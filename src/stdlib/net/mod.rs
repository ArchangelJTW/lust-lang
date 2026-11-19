#![cfg(feature = "std")]

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::config::LustConfig;
use crate::vm::VM;

static NEXT_NET_HANDLE: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static TCP_LISTENERS: RefCell<HashMap<u64, TcpListener>> = RefCell::new(HashMap::new());
    static TCP_STREAMS: RefCell<HashMap<u64, TcpStream>> = RefCell::new(HashMap::new());
}

pub fn reset_global_net_registries() {
    TCP_LISTENERS.with(|m| m.borrow_mut().clear());
    TCP_STREAMS.with(|m| m.borrow_mut().clear());
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetError {
    CapabilityDenied(String),
    Timeout(String),
    Cancelled(String),
    InvalidHandle(u64),
    Io(String),
}

impl std::fmt::Display for NetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapabilityDenied(msg) => write!(f, "CapabilityDenied: {msg}"),
            Self::Timeout(msg) => write!(f, "Timeout: {msg}"),
            Self::Cancelled(msg) => write!(f, "Cancelled: {msg}"),
            Self::InvalidHandle(id) => write!(f, "InvalidHandle: unknown net handle {id}"),
            Self::Io(msg) => write!(f, "NetIoError: {msg}"),
        }
    }
}

impl std::error::Error for NetError {}

pub fn check_net_capability(config: &LustConfig, module_name: &str) -> Result<(), NetError> {
    if !config.capability_mask().allow_net {
        return Err(NetError::CapabilityDenied(format!(
            "network capability denied for '{module_name}' (allow_net = false)"
        )));
    }
    if !config.is_std_module_allowed(module_name)
        && !config.capability_mask().custom_capabilities.contains(module_name)
    {
        return Err(NetError::CapabilityDenied(format!(
            "module '{module_name}' is not allowed in current capability mask"
        )));
    }
    Ok(())
}

pub fn check_vm_net_capability(vm: &VM, module_name: &str) -> Result<(), NetError> {
    check_net_capability(vm.config(), module_name)
}

#[derive(Debug, Clone)]
pub struct CancelToken {
    cancelled: Arc<AtomicBool>,
    #[cfg(feature = "net-services")]
    notify: Arc<tokio::sync::Notify>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "net-services")]
            notify: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        #[cfg(feature = "net-services")]
        self.notify.notify_waiters();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    pub fn ensure_active(&self) -> Result<(), NetError> {
        if self.is_cancelled() {
            Err(NetError::Cancelled("operation cancelled".to_string()))
        } else {
            Ok(())
        }
    }

    #[cfg(feature = "net-services")]
    pub async fn cancelled(&self) {
        loop {
            let notified = self.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Core TCP loopback and bounded-socket service (`std.net` / `@std/net`).
pub struct NetService;

impl NetService {
    pub fn tcp_listen(config: &LustConfig, host: &str, port: u16) -> Result<(u64, SocketAddr), NetError> {
        check_net_capability(config, "net")?;
        let listener = TcpListener::bind((host, port))
            .map_err(|e| NetError::Io(format!("bind {host}:{port} failed: {e}")))?;
        let addr = listener
            .local_addr()
            .map_err(|e| NetError::Io(format!("local_addr failed: {e}")))?;
        let id = NEXT_NET_HANDLE.fetch_add(1, Ordering::Relaxed);
        TCP_LISTENERS.with(|m| {
            m.borrow_mut().insert(id, listener);
        });
        Ok((id, addr))
    }

    pub fn tcp_accept(
        config: &LustConfig,
        listener_id: u64,
        timeout: Option<Duration>,
        cancel: Option<&CancelToken>,
    ) -> Result<(u64, SocketAddr), NetError> {
        check_net_capability(config, "net")?;
        if let Some(tok) = cancel {
            tok.ensure_active()?;
        }
        let cloned = TCP_LISTENERS.with(|m| {
            m.borrow()
                .get(&listener_id)
                .and_then(|l| l.try_clone().ok())
                .ok_or(NetError::InvalidHandle(listener_id))
        })?;
        cloned
            .set_nonblocking(true)
            .map_err(|e| NetError::Io(format!("set_nonblocking failed: {e}")))?;

        let deadline = timeout.map(|d| std::time::Instant::now() + d);
        loop {
            if let Some(tok) = cancel {
                tok.ensure_active()?;
            }
            match cloned.accept() {
                Ok((stream, peer)) => {
                    let _ = stream.set_nonblocking(false);
                    let id = NEXT_NET_HANDLE.fetch_add(1, Ordering::Relaxed);
                    TCP_STREAMS.with(|m| {
                        m.borrow_mut().insert(id, stream);
                    });
                    return Ok((id, peer));
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if let Some(dl) = deadline {
                        if std::time::Instant::now() >= dl {
                            return Err(NetError::Timeout(format!(
                                "tcp_accept timed out on listener {listener_id}"
                            )));
                        }
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(e) => return Err(NetError::Io(format!("accept failed: {e}"))),
            }
        }
    }

    pub fn tcp_connect(
        config: &LustConfig,
        host: &str,
        port: u16,
        timeout: Duration,
        cancel: Option<&CancelToken>,
    ) -> Result<u64, NetError> {
        check_net_capability(config, "net")?;
        if let Some(tok) = cancel {
            tok.ensure_active()?;
        }
        let mut addrs = (host, port)
            .to_socket_addrs()
            .map_err(|e| NetError::Io(format!("resolve {host}:{port} failed: {e}")))?;
        let addr = addrs
            .next()
            .ok_or_else(|| NetError::Io(format!("no socket addresses resolved for {host}:{port}")))?;
        let stream = TcpStream::connect_timeout(&addr, timeout).map_err(|e| {
            if e.kind() == std::io::ErrorKind::TimedOut {
                NetError::Timeout(format!("connect to {addr} timed out"))
            } else {
                NetError::Io(format!("connect to {addr} failed: {e}"))
            }
        })?;
        let _ = stream.set_read_timeout(Some(timeout));
        let _ = stream.set_write_timeout(Some(timeout));
        let id = NEXT_NET_HANDLE.fetch_add(1, Ordering::Relaxed);
        TCP_STREAMS.with(|m| {
            m.borrow_mut().insert(id, stream);
        });
        Ok(id)
    }

    pub fn tcp_write_all(
        config: &LustConfig,
        stream_id: u64,
        data: &[u8],
    ) -> Result<usize, NetError> {
        check_net_capability(config, "net")?;
        TCP_STREAMS.with(|m| {
            let mut guard = m.borrow_mut();
            let stream = guard
                .get_mut(&stream_id)
                .ok_or(NetError::InvalidHandle(stream_id))?;
            stream
                .write_all(data)
                .map_err(|e| NetError::Io(format!("write failed: {e}")))?;
            stream
                .flush()
                .map_err(|e| NetError::Io(format!("flush failed: {e}")))?;
            Ok(data.len())
        })
    }

    pub fn tcp_read_exact_or_eof(
        config: &LustConfig,
        stream_id: u64,
        max_bytes: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<u8>, NetError> {
        check_net_capability(config, "net")?;
        TCP_STREAMS.with(|m| {
            let mut guard = m.borrow_mut();
            let stream = guard
                .get_mut(&stream_id)
                .ok_or(NetError::InvalidHandle(stream_id))?;
            if let Some(t) = timeout {
                let _ = stream.set_read_timeout(Some(t));
            }
            let mut buf = vec![0u8; max_bytes];
            match stream.read(&mut buf) {
                Ok(n) => {
                    buf.truncate(n);
                    Ok(buf)
                }
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    Err(NetError::Timeout(format!(
                        "tcp_read timed out on stream {stream_id}"
                    )))
                }
                Err(e) => Err(NetError::Io(format!("read failed: {e}"))),
            }
        })
    }

    pub fn tcp_close_stream(stream_id: u64) -> bool {
        TCP_STREAMS.with(|m| m.borrow_mut().remove(&stream_id).is_some())
    }

    pub fn tcp_close_listener(listener_id: u64) -> bool {
        TCP_LISTENERS.with(|m| m.borrow_mut().remove(&listener_id).is_some())
    }

    pub fn active_handle_counts() -> (usize, usize) {
        let listeners = TCP_LISTENERS.with(|m| m.borrow().len());
        let streams = TCP_STREAMS.with(|m| m.borrow().len());
        (listeners, streams)
    }
}

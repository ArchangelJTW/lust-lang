#![cfg(feature = "std")]

#[cfg(feature = "net-services")]
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use std::collections::HashMap;
#[cfg(feature = "net-services")]
use std::net::SocketAddr;
#[cfg(feature = "net-services")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "net-services")]
use std::time::Duration;

use crate::bytecode::{LustMap, NativeCallResult, Value, ValueKey, native_fn};
#[cfg(feature = "net-services")]
use crate::config::LustConfig;
#[cfg(feature = "net-services")]
use crate::stdlib::net::{CancelToken, NetError, check_net_capability};
use crate::stdlib::net::{NetService, reset_global_net_registries};
use crate::vm::VM;

#[cfg(feature = "net-services")]
static NEXT_DB_HANDLE: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "net-services")]
thread_local! {
    static SQLITE_CONNS: RefCell<HashMap<u64, rusqlite::Connection>> = RefCell::new(HashMap::new());
}

#[cfg(not(feature = "net-services"))]
thread_local! {
    static SQLITE_CONNS: RefCell<HashMap<u64, ()>> = RefCell::new(HashMap::new());
}

pub fn reset_global_service_registries() {
    reset_global_net_registries();
    SQLITE_CONNS.with(|m| m.borrow_mut().clear());
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
    pub headers: Vec<(String, String)>,
}

/// Real `hyper` + `tokio` loopback HTTP/1.1 service (`reqwest` remains BLOCKED offline).
pub struct HttpService;

#[cfg(feature = "net-services")]
pub struct LoopbackHttpServer {
    addr: SocketAddr,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(feature = "net-services")]
impl LoopbackHttpServer {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(feature = "net-services")]
impl Drop for LoopbackHttpServer {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(feature = "net-services")]
impl HttpService {
    /// Starts an isolated `tokio` + `hyper 1.12` HTTP/1.1 loopback test/local server on `127.0.0.1:0`.
    pub fn start_loopback_server<F>(
        config: &LustConfig,
        handler: F,
    ) -> Result<LoopbackHttpServer, NetError>
    where
        F: Fn(String, String, Vec<u8>) -> (u16, Vec<u8>) + Send + Sync + 'static,
    {
        use bytes::Bytes;
        use http_body_util::{BodyExt, Full};
        use hyper::server::conn::http1;
        use hyper::service::service_fn;
        use hyper::{Request, Response, StatusCode};
        use hyper_util::rt::TokioIo;
        use std::sync::Arc;

        check_net_capability(config, "http")?;

        let std_listener = std::net::TcpListener::bind("127.0.0.1:0")
            .map_err(|e| NetError::Io(format!("loopback http bind failed: {e}")))?;
        std_listener
            .set_nonblocking(true)
            .map_err(|e| NetError::Io(format!("set_nonblocking failed: {e}")))?;
        let addr = std_listener
            .local_addr()
            .map_err(|e| NetError::Io(format!("local_addr failed: {e}")))?;

        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let handler = Arc::new(handler);

        let join_handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio current_thread runtime for loopback hyper");
            rt.block_on(async move {
                let listener = tokio::net::TcpListener::from_std(std_listener)
                    .expect("tokio TcpListener from std");
                loop {
                    tokio::select! {
                        _ = &mut shutdown_rx => {
                            break;
                        }
                        accepted = listener.accept() => {
                            match accepted {
                                Ok((stream, _)) => {
                                    let io = TokioIo::new(stream);
                                    let h = Arc::clone(&handler);
                                    tokio::spawn(async move {
                                        let svc = service_fn(move |req: Request<hyper::body::Incoming>| {
                                            let h = Arc::clone(&h);
                                            async move {
                                                let method = req.method().as_str().to_string();
                                                let path = req.uri().path_and_query().map(|p| p.as_str()).unwrap_or("/").to_string();
                                                let body_bytes = match req.into_body().collect().await {
                                                    Ok(col) => col.to_bytes().to_vec(),
                                                    Err(_) => Vec::new(),
                                                };
                                                let (status_u16, resp_bytes) = h(method, path, body_bytes);
                                                let status = StatusCode::from_u16(status_u16).unwrap_or(StatusCode::OK);
                                                let resp = Response::builder()
                                                    .status(status)
                                                    .header("x-lust-runtime", "hyper-1.12-loopback")
                                                    .body(Full::new(Bytes::from(resp_bytes)))
                                                    .unwrap();
                                                Ok::<_, !>(resp)
                                            }
                                        });
                                        let _ = http1::Builder::new().serve_connection(io, svc).await;
                                    });
                                }
                                Err(_) => break,
                            }
                        }
                    }
                }
            });
        });

        Ok(LoopbackHttpServer {
            addr,
            shutdown_tx: Some(shutdown_tx),
            join_handle: Some(join_handle),
        })
    }

    pub fn request(
        config: &LustConfig,
        method: &str,
        url: &str,
        body: &[u8],
        timeout: Duration,
        cancel: Option<&CancelToken>,
    ) -> Result<HttpResponse, NetError> {
        use bytes::Bytes;
        use http_body_util::{BodyExt, Full};
        use hyper::{Method, Request};
        use hyper_util::rt::TokioIo;

        check_net_capability(config, "http")?;
        if let Some(tok) = cancel {
            tok.ensure_active()?;
        }

        let uri: hyper::Uri = url
            .parse()
            .map_err(|e| NetError::Io(format!("invalid url '{url}': {e}")))?;
        let host = uri
            .host()
            .ok_or_else(|| NetError::Io(format!("missing host in url '{url}'")))?
            .to_string();
        let port = uri.port_u16().unwrap_or(80);
        let parsed_method = Method::from_bytes(method.as_bytes())
            .map_err(|e| NetError::Io(format!("invalid HTTP method '{method}': {e}")))?;
        let path_pq = uri
            .path_and_query()
            .map(|p| p.as_str().to_string())
            .unwrap_or_else(|| "/".to_string());

        let start_deadline = std::time::Instant::now() + timeout;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| NetError::Io(format!("tokio runtime build failed: {e}")))?;

        if std::time::Instant::now() >= start_deadline {
            return Err(NetError::Timeout(format!(
                "HTTP {method} {url} timed out during runtime setup ({}ms)",
                timeout.as_millis()
            )));
        }

        let body_vec = body.to_vec();
        let tokio_deadline = tokio::time::Instant::from_std(start_deadline);
        rt.block_on(async move {
            let cancel_fut = async move {
                match cancel {
                    Some(tok) => tok.cancelled().await,
                    None => core::future::pending::<()>().await,
                }
            };

            let req_future = async move {
                if let Some(tok) = cancel {
                    tok.ensure_active()?;
                }
                let stream = tokio::net::TcpStream::connect((host.as_str(), port))
                    .await
                    .map_err(|e| NetError::Io(format!("http connect failed: {e}")))?;
                let io = TokioIo::new(stream);
                let (mut sender, conn) = hyper::client::conn::http1::handshake(io)
                    .await
                    .map_err(|e| NetError::Io(format!("http1 handshake failed: {e}")))?;
                let conn_task = tokio::spawn(async move {
                    let _ = conn.await;
                });

                if let Some(tok) = cancel {
                    tok.ensure_active()?;
                }

                let req = Request::builder()
                    .method(parsed_method)
                    .uri(path_pq)
                    .header("host", format!("{host}:{port}"))
                    .header("content-length", body_vec.len().to_string())
                    .body(Full::new(Bytes::from(body_vec)))
                    .map_err(|e| NetError::Io(format!("build request failed: {e}")))?;

                let res = sender
                    .send_request(req)
                    .await
                    .map_err(|e| NetError::Io(format!("send_request failed: {e}")))?;
                let status = res.status().as_u16();
                let headers = res
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect::<Vec<_>>();
                let collected = res
                    .into_body()
                    .collect()
                    .await
                    .map_err(|e| NetError::Io(format!("read response body failed: {e}")))?;
                conn_task.abort();
                Ok(HttpResponse {
                    status,
                    body: collected.to_bytes().to_vec(),
                    headers,
                })
            };

            tokio::select! {
                _ = cancel_fut => Err(NetError::Cancelled(format!(
                    "HTTP {method} {url} cancelled while in flight"
                ))),
                timed = tokio::time::timeout_at(tokio_deadline, req_future) => match timed {
                    Ok(inner) => inner,
                    Err(_) => Err(NetError::Timeout(format!(
                        "HTTP {method} {url} timed out after {}ms",
                        timeout.as_millis()
                    ))),
                }
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SqlValue {
    Null,
    Int(i64),
    Float(f64),
    Text(String),
}

/// Real `rusqlite` SQLite service (`sqlx` remains BLOCKED offline).
pub struct SqliteService;

#[cfg(feature = "net-services")]
impl SqliteService {
    pub fn check_capability(config: &LustConfig, path: &str) -> Result<(), NetError> {
        if !config.is_std_module_allowed("sqlite")
            && !config.capability_mask().custom_capabilities.contains("sqlite")
        {
            return Err(NetError::CapabilityDenied(
                "module 'sqlite' is not allowed in capability mask".to_string(),
            ));
        }
        if path != ":memory:" && !config.capability_mask().allow_fs {
            return Err(NetError::CapabilityDenied(format!(
                "file-backed sqlite path '{path}' requires allow_fs capability"
            )));
        }
        Ok(())
    }

    pub fn open(config: &LustConfig, path: &str) -> Result<u64, NetError> {
        Self::check_capability(config, path)?;
        let conn = if path == ":memory:" {
            rusqlite::Connection::open_in_memory()
        } else {
            rusqlite::Connection::open(path)
        }
        .map_err(|e| NetError::Io(format!("sqlite open '{path}' failed: {e}")))?;
        let id = NEXT_DB_HANDLE.fetch_add(1, Ordering::Relaxed);
        SQLITE_CONNS.with(|m| {
            m.borrow_mut().insert(id, conn);
        });
        Ok(id)
    }

    pub fn execute_batch(db_id: u64, sql: &str) -> Result<(), NetError> {
        SQLITE_CONNS.with(|m| {
            let guard = m.borrow();
            let conn = guard.get(&db_id).ok_or(NetError::InvalidHandle(db_id))?;
            conn.execute_batch(sql)
                .map_err(|e| NetError::Io(format!("sqlite execute_batch failed: {e}")))
        })
    }

    pub fn execute_params(db_id: u64, sql: &str, params: &[SqlValue]) -> Result<usize, NetError> {
        SQLITE_CONNS.with(|m| {
            let guard = m.borrow();
            let conn = guard.get(&db_id).ok_or(NetError::InvalidHandle(db_id))?;
            let boxed: Vec<rusqlite::types::Value> = params.iter().map(to_rusqlite_val).collect();
            conn.execute(sql, rusqlite::params_from_iter(boxed.iter()))
                .map_err(|e| NetError::Io(format!("sqlite execute failed: {e}")))
        })
    }

    pub fn query_rows(
        db_id: u64,
        sql: &str,
        params: &[SqlValue],
    ) -> Result<Vec<Vec<SqlValue>>, NetError> {
        SQLITE_CONNS.with(|m| {
            let guard = m.borrow();
            let conn = guard.get(&db_id).ok_or(NetError::InvalidHandle(db_id))?;
            let mut stmt = conn
                .prepare_cached(sql)
                .map_err(|e| NetError::Io(format!("sqlite prepare failed: {e}")))?;
            let col_count = stmt.column_count();
            let boxed: Vec<rusqlite::types::Value> = params.iter().map(to_rusqlite_val).collect();
            let rows_iter = stmt
                .query_map(rusqlite::params_from_iter(boxed.iter()), |row| {
                    let mut out = Vec::with_capacity(col_count);
                    for idx in 0..col_count {
                        let v: rusqlite::types::Value = row.get(idx)?;
                        out.push(from_rusqlite_val(v));
                    }
                    Ok(out)
                })
                .map_err(|e| NetError::Io(format!("sqlite query failed: {e}")))?;

            let mut results = Vec::new();
            for row in rows_iter {
                results.push(row.map_err(|e| NetError::Io(format!("sqlite row decode failed: {e}")))?);
            }
            Ok(results)
        })
    }

    pub fn close(db_id: u64) -> bool {
        SQLITE_CONNS.with(|m| m.borrow_mut().remove(&db_id).is_some())
    }

    pub fn active_connection_count() -> usize {
        SQLITE_CONNS.with(|m| m.borrow().len())
    }
}

#[cfg(feature = "net-services")]
fn to_rusqlite_val(v: &SqlValue) -> rusqlite::types::Value {
    match v {
        SqlValue::Null => rusqlite::types::Value::Null,
        SqlValue::Int(i) => rusqlite::types::Value::Integer(*i),
        SqlValue::Float(f) => rusqlite::types::Value::Real(*f),
        SqlValue::Text(s) => rusqlite::types::Value::Text(s.clone()),
    }
}

#[cfg(feature = "net-services")]
fn from_rusqlite_val(v: rusqlite::types::Value) -> SqlValue {
    match v {
        rusqlite::types::Value::Null => SqlValue::Null,
        rusqlite::types::Value::Integer(i) => SqlValue::Int(i),
        rusqlite::types::Value::Real(f) => SqlValue::Float(f),
        rusqlite::types::Value::Text(s) => SqlValue::Text(s),
        rusqlite::types::Value::Blob(b) => SqlValue::Text(format!("<blob:{}>", b.len())),
    }
}

fn str_key(s: &str) -> ValueKey {
    ValueKey::string(s.to_string())
}

/// Registers `@std/net`, `@std/http`, `@std/sqlite`, and `@std/services` native module namespaces onto a VM.
pub fn register_builtins(vm: &mut VM) {
    let config_snapshot = vm.config().clone();

    let cfg_listen = config_snapshot.clone();
    let net_listen = native_fn(move |args: &[Value]| -> Result<NativeCallResult, String> {
        let host = args
            .first()
            .and_then(|v| v.as_string())
            .unwrap_or("127.0.0.1");
        let port = args.get(1).and_then(|v| v.as_int()).unwrap_or(0) as u16;
        match NetService::tcp_listen(&cfg_listen, host, port) {
            Ok((id, addr)) => {
                let mut m = LustMap::default();
                m.insert(str_key("id"), Value::Int(id as i64));
                m.insert(str_key("port"), Value::Int(addr.port() as i64));
                m.insert(str_key("addr"), Value::string(addr.to_string()));
                Ok(NativeCallResult::Return(Value::map(m)))
            }
            Err(e) => Err(e.to_string()),
        }
    });

    let net_close_listener = native_fn(move |args: &[Value]| -> Result<NativeCallResult, String> {
        let id = args.first().and_then(|v| v.as_int()).unwrap_or(-1);
        Ok(NativeCallResult::Return(Value::Bool(
            id > 0 && NetService::tcp_close_listener(id as u64),
        )))
    });

    let blocked_probe = native_fn(move |_args: &[Value]| -> Result<NativeCallResult, String> {
        let mut m = LustMap::default();
        m.insert(str_key("reqwest"), Value::string("BLOCKED: crate missing offline"));
        m.insert(str_key("sqlx"), Value::string("BLOCKED: crate missing offline"));
        m.insert(str_key("http_backend"), Value::string("hyper-1.12-loopback"));
        m.insert(str_key("sqlite_backend"), Value::string("rusqlite-0.33"));
        Ok(NativeCallResult::Return(Value::map(m)))
    });

    let mut net_map = LustMap::default();
    net_map.insert(str_key("listen"), Value::NativeFunction(net_listen.clone()));
    net_map.insert(
        str_key("close_listener"),
        Value::NativeFunction(net_close_listener.clone()),
    );
    let net_val = Value::map(net_map);
    vm.register_native("net", net_val.clone());
    vm.register_native("std.net", net_val.clone());
    vm.register_native("@std/net", net_val);
    vm.register_native("net.listen", Value::NativeFunction(net_listen));
    vm.register_native(
        "net.close_listener",
        Value::NativeFunction(net_close_listener),
    );

    let mut svc_map = LustMap::default();
    svc_map.insert(str_key("backend_status"), Value::NativeFunction(blocked_probe.clone()));
    #[cfg(feature = "net-services")]
    {
        let cfg_http = config_snapshot.clone();
        let http_get = native_fn(move |args: &[Value]| -> Result<NativeCallResult, String> {
            let url = args
                .first()
                .and_then(|v| v.as_string())
                .ok_or_else(|| "http.get(url, timeout_ms?) requires url string".to_string())?;
            let timeout_ms = args.get(1).and_then(|v| v.as_int()).unwrap_or(2000).max(1) as u64;
            match HttpService::request(&cfg_http, "GET", url, &[], Duration::from_millis(timeout_ms), None) {
                Ok(resp) => {
                    let mut m = LustMap::default();
                    m.insert(str_key("status"), Value::Int(resp.status as i64));
                    m.insert(
                        str_key("body"),
                        Value::string(String::from_utf8_lossy(&resp.body).into_owned()),
                    );
                    Ok(NativeCallResult::Return(Value::map(m)))
                }
                Err(e) => Err(e.to_string()),
            }
        });
        svc_map.insert(str_key("http_get"), Value::NativeFunction(http_get.clone()));
        let mut http_map = LustMap::default();
        http_map.insert(str_key("get"), Value::NativeFunction(http_get.clone()));
        let http_val = Value::map(http_map);
        vm.register_native("http", http_val.clone());
        vm.register_native("std.http", http_val.clone());
        vm.register_native("@std/http", http_val);
        vm.register_native("http.get", Value::NativeFunction(http_get));

        let cfg_sql = config_snapshot;
        let sql_open = native_fn(move |args: &[Value]| -> Result<NativeCallResult, String> {
            let path = args.first().and_then(|v| v.as_string()).unwrap_or(":memory:");
            SqliteService::open(&cfg_sql, path)
                .map(|id| NativeCallResult::Return(Value::Int(id as i64)))
                .map_err(|e| e.to_string())
        });
        let sql_close = native_fn(move |args: &[Value]| -> Result<NativeCallResult, String> {
            let id = args.first().and_then(|v| v.as_int()).unwrap_or(-1);
            Ok(NativeCallResult::Return(Value::Bool(
                id > 0 && SqliteService::close(id as u64),
            )))
        });
        let mut sqlite_map = LustMap::default();
        sqlite_map.insert(str_key("open"), Value::NativeFunction(sql_open.clone()));
        sqlite_map.insert(str_key("close"), Value::NativeFunction(sql_close.clone()));
        let sqlite_val = Value::map(sqlite_map);
        vm.register_native("sqlite", sqlite_val.clone());
        vm.register_native("std.sqlite", sqlite_val.clone());
        vm.register_native("@std/sqlite", sqlite_val);
        vm.register_native("sqlite.open", Value::NativeFunction(sql_open));
        vm.register_native("sqlite.close", Value::NativeFunction(sql_close));
    }

    let svc_val = Value::map(svc_map);
    vm.register_native("services", svc_val.clone());
    vm.register_native("std.services", svc_val.clone());
    vm.register_native("@std/services", svc_val);
    vm.register_native("services.backend_status", Value::NativeFunction(blocked_probe));
}

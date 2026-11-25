#![cfg(feature = "std")]

use std::time::Duration;

use lust::config::{LustConfig, PreludeMode};
use lust::stdlib::net::{CancelToken, NetError, NetService};
use lust::stdlib::services::{HttpService, SqlValue, SqliteService};
use lust::stdlib::services::register_builtins as register_service_builtins;
use lust::{Value, VM};
use tempfile::TempDir;

#[test]
fn w4_4b_tcp_loopback_roundtrip_timeout_cancel_and_cleanup() {
    let config = LustConfig::default();
    let (listener_id, addr) = NetService::tcp_listen(&config, "127.0.0.1", 0).expect("tcp listen");

    let cfg_client = config.clone();
    let client_handle = std::thread::spawn(move || {
        let sid = NetService::tcp_connect(
            &cfg_client,
            "127.0.0.1",
            addr.port(),
            Duration::from_secs(2),
            None,
        )
        .expect("tcp connect");
        NetService::tcp_write_all(&cfg_client, sid, b"PING-W4-4B").expect("tcp write");
        let reply = NetService::tcp_read_exact_or_eof(&cfg_client, sid, 64, Some(Duration::from_secs(2)))
            .expect("tcp read reply");
        assert!(NetService::tcp_close_stream(sid));
        reply
    });

    let (srv_sid, _peer) = NetService::tcp_accept(
        &config,
        listener_id,
        Some(Duration::from_secs(2)),
        None,
    )
    .expect("tcp accept");
    let got = NetService::tcp_read_exact_or_eof(&config, srv_sid, 64, Some(Duration::from_secs(2)))
        .expect("server read");
    assert_eq!(got, b"PING-W4-4B");
    NetService::tcp_write_all(&config, srv_sid, b"PONG-W4-4B").expect("server write");
    assert!(NetService::tcp_close_stream(srv_sid));
    assert!(NetService::tcp_close_listener(listener_id));

    let reply = client_handle.join().expect("client join");
    assert_eq!(reply, b"PONG-W4-4B");
    assert_eq!(NetService::active_handle_counts(), (0, 0));

    // Cancellation & timeout oracle on idle listener
    let (idle_lid, _) = NetService::tcp_listen(&config, "127.0.0.1", 0).expect("idle listen");
    let cancel = CancelToken::new();
    cancel.cancel();
    let cancelled = NetService::tcp_accept(&config, idle_lid, Some(Duration::from_secs(1)), Some(&cancel));
    assert!(matches!(cancelled, Err(NetError::Cancelled(_))));

    let timed_out = NetService::tcp_accept(&config, idle_lid, Some(Duration::from_millis(25)), None);
    assert!(matches!(timed_out, Err(NetError::Timeout(_))));
    assert!(NetService::tcp_close_listener(idle_lid));
    assert_eq!(NetService::active_handle_counts(), (0, 0));
}

#[test]
fn w4_4b_capability_denial_blocks_net_http_and_sqlite() {
    let mut no_net_cfg = LustConfig::default();
    no_net_cfg.set_strict_capabilities(true);
    no_net_cfg.capability_mask_mut().allow_net = false;
    no_net_cfg.capability_mask_mut().allow_fs = false;

    let err_tcp = NetService::tcp_listen(&no_net_cfg, "127.0.0.1", 0).expect_err("tcp denied");
    assert!(matches!(err_tcp, NetError::CapabilityDenied(_)));

    let err_http = HttpService::request(
        &no_net_cfg,
        "GET",
        "http://127.0.0.1:1/",
        &[],
        Duration::from_millis(100),
        None,
    )
    .expect_err("http denied");
    assert!(matches!(err_http, NetError::CapabilityDenied(_)));

    let err_sql = SqliteService::open(&no_net_cfg, ":memory:").expect_err("sqlite denied");
    assert!(matches!(err_sql, NetError::CapabilityDenied(_)));

    // Native VM binding denial check
    let mut denied_cfg = LustConfig::default();
    denied_cfg.set_prelude_mode(PreludeMode::Pure);
    denied_cfg.set_strict_capabilities(true);
    denied_cfg.capability_mask_mut().allow_net = false;
    let mut vm = VM::with_config(&denied_cfg);
    register_service_builtins(&mut vm);
    let native_fn = vm
        .get_global("net.listen")
        .expect("net.listen global registered");
    if let Value::NativeFunction(f) = native_fn {
        let err = f(&[Value::string("127.0.0.1"), Value::Int(0)]).expect_err("must reject allow_net=false");
        assert!(format!("{err}").contains("CapabilityDenied"));
    } else {
        panic!("expected NativeFunction for net.listen");
    }
}

#[test]
fn w4_4b_hyper_http_loopback_get_post_timeout_and_cancel() {
    let config = LustConfig::default();
    let mut server = HttpService::start_loopback_server(&config, |method, path, body| {
        if path == "/slow" {
            std::thread::sleep(Duration::from_millis(250));
            return (200, b"too-late".to_vec());
        }
        if method == "POST" && path == "/echo" {
            return (201, body);
        }
        (200, format!("ok:{method}:{path}").into_bytes())
    })
    .expect("start loopback hyper server");

    let base = server.base_url();
    let get_resp = HttpService::request(
        &config,
        "GET",
        &format!("{base}/health?v=1"),
        &[],
        Duration::from_secs(2),
        None,
    )
    .expect("HTTP GET");
    assert_eq!(get_resp.status, 200);
    assert_eq!(get_resp.body, b"ok:GET:/health?v=1");

    let post_resp = HttpService::request(
        &config,
        "POST",
        &format!("{base}/echo"),
        br#"{"msg":"hello-hyper"}"#,
        Duration::from_secs(2),
        None,
    )
    .expect("HTTP POST");
    assert_eq!(post_resp.status, 201);
    assert_eq!(post_resp.body, br#"{"msg":"hello-hyper"}"#);

    let timeout_err = HttpService::request(
        &config,
        "GET",
        &format!("{base}/slow"),
        &[],
        Duration::from_millis(40),
        None,
    )
    .expect_err("slow endpoint must time out");
    assert!(matches!(timeout_err, NetError::Timeout(_)));

    let cancel = CancelToken::new();
    cancel.cancel();
    let cancel_err = HttpService::request(
        &config,
        "GET",
        &format!("{base}/health"),
        &[],
        Duration::from_secs(2),
        Some(&cancel),
    )
    .expect_err("cancelled HTTP request must abort cleanly");
    assert!(matches!(cancel_err, NetError::Cancelled(_)));

    // Active in-flight HTTP cancellation: abort /slow well before its 250ms handler delay
    let active_cancel = CancelToken::new();
    let active_cancel_bg = active_cancel.clone();
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(30));
        active_cancel_bg.cancel();
    });
    let t0 = std::time::Instant::now();
    let in_flight_err = HttpService::request(
        &config,
        "GET",
        &format!("{base}/slow"),
        &[],
        Duration::from_secs(2),
        Some(&active_cancel),
    )
    .expect_err("in-flight HTTP /slow must cancel cleanly when token fires");
    let elapsed = t0.elapsed();
    canceller.join().unwrap();
    assert!(
        matches!(in_flight_err, NetError::Cancelled(_)),
        "expected NetError::Cancelled, got {in_flight_err:?}"
    );
    assert!(
        elapsed < Duration::from_millis(200),
        "in-flight cancellation took too long: {elapsed:?}"
    );

    server.stop();
}

#[test]
fn w4_4b_sqlite_rusqlite_crud_transactions_injection_safety_and_close() {
    let config = LustConfig::default();
    let tmp = TempDir::new().expect("tempdir");
    let db_path = tmp.path().join("w4_4b_test.sqlite3");

    let db_id = SqliteService::open(&config, db_path.to_str().unwrap()).expect("sqlite open");
    SqliteService::execute_batch(
        db_id,
        "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL, qty INTEGER NOT NULL);",
    )
    .expect("create table");

    // Parameterized insert & injection-payload literal binding
    let n = SqliteService::execute_params(
        db_id,
        "INSERT INTO items (id, name, qty) VALUES (?1, ?2, ?3)",
        &[
            SqlValue::Int(1),
            SqlValue::Text("widget' OR '1'='1".to_string()),
            SqlValue::Int(42),
        ],
    )
    .expect("insert row 1");
    assert_eq!(n, 1);

    // Transaction rollback leaves state untouched
    SqliteService::execute_batch(db_id, "BEGIN TRANSACTION;").expect("begin");
    SqliteService::execute_params(
        db_id,
        "INSERT INTO items (id, name, qty) VALUES (?1, ?2, ?3)",
        &[
            SqlValue::Int(2),
            SqlValue::Text("rolled-back".to_string()),
            SqlValue::Int(99),
        ],
    )
    .expect("insert row 2");
    SqliteService::execute_batch(db_id, "ROLLBACK;").expect("rollback");

    let rows = SqliteService::query_rows(
        db_id,
        "SELECT id, name, qty FROM items WHERE name = ?1",
        &[SqlValue::Text("widget' OR '1'='1".to_string())],
    )
    .expect("query literal injection key");
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0],
        vec![
            SqlValue::Int(1),
            SqlValue::Text("widget' OR '1'='1".to_string()),
            SqlValue::Int(42),
        ]
    );

    let all_rows = SqliteService::query_rows(db_id, "SELECT COUNT(*) FROM items", &[]).unwrap();
    assert_eq!(all_rows, vec![vec![SqlValue::Int(1)]]);

    assert!(SqliteService::close(db_id));
    assert_eq!(SqliteService::active_connection_count(), 0);
}

#[test]
fn w4_4b_explicit_blocked_external_crates_and_vm_backend_status() {
    let mut vm = VM::new();
    register_service_builtins(&mut vm);
    let status_fn = vm
        .get_global("services.backend_status")
        .expect("services.backend_status");
    if let Value::NativeFunction(f) = status_fn {
        match f(&[]).expect("call backend_status") {
            lust::bytecode::NativeCallResult::Return(Value::Map(m)) => {
                let borrow = m.borrow();
                assert!(format!("{borrow:?}").contains("reqwest"));
                assert!(format!("{borrow:?}").contains("BLOCKED"));
                assert!(format!("{borrow:?}").contains("sqlx"));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    } else {
        panic!("expected NativeFunction");
    }
}

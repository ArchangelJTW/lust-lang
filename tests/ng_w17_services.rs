#[cfg(all(feature = "net-services", feature = "svc-reqwest"))]
#[path = "../src/stdlib/services/reqwest_adapter.rs"]
mod reqwest_adapter;
#[cfg(all(feature = "net-services", feature = "svc-sqlx"))]
#[path = "../src/stdlib/services/sqlx_adapter.rs"]
mod sqlx_adapter;

#[cfg(all(
    feature = "net-services",
    any(feature = "svc-reqwest", feature = "svc-sqlx")
))]
use lust::config::LustConfig;
#[cfg(all(feature = "net-services", feature = "svc-reqwest"))]
use lust::stdlib::net::CancelToken;
#[cfg(all(
    feature = "net-services",
    any(feature = "svc-reqwest", feature = "svc-sqlx")
))]
use std::time::Duration;

#[cfg(all(feature = "net-services", feature = "svc-reqwest"))]
#[tokio::test]
async fn adapters_deny_before_connecting() {
    let mut config = LustConfig::default();
    config.capability_mask_mut().allow_net = false;
    let denied = reqwest_adapter::ReqwestAdapter::get(&config, "http://127.0.0.1:1/", Duration::from_secs(1), &CancelToken::new()).await;
    assert!(matches!(denied, Err(lust::stdlib::net::NetError::CapabilityDenied(_))));
}

#[cfg(all(feature = "net-services", feature = "svc-reqwest"))]
#[tokio::test]
async fn reqwest_uses_deterministic_loopback() {
    use std::{io::{Read, Write}, net::TcpListener, thread};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0; 1024]; let _ = stream.read(&mut buf);
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok").unwrap();
    });
    let config = LustConfig::default();
    let (status, body) = reqwest_adapter::ReqwestAdapter::get(&config, &format!("http://{addr}/"), Duration::from_secs(2), &CancelToken::new()).await.unwrap();
    assert_eq!(status, 200); assert_eq!(body, b"ok"); server.join().unwrap();
}

#[cfg(all(feature = "net-services", feature = "svc-sqlx"))]
#[tokio::test]
async fn sqlx_memory_database_supports_bound_values() {
    let config = LustConfig::default();
    let pool = sqlx_adapter::SqlxAdapter::connect(&config, "sqlite::memory:", Duration::from_secs(2)).await.unwrap();
    sqlx::query("CREATE TABLE t (v TEXT)").execute(&pool).await.unwrap();
    assert_eq!(sqlx_adapter::SqlxAdapter::execute(&pool, "INSERT INTO t VALUES (?)", "safe").await.unwrap(), 1);
    pool.close().await;
}

#[cfg(all(feature = "net-services", feature = "svc-sqlx"))]
#[tokio::test]
async fn sqlx_transaction_rolls_back_on_error() {
    let config = LustConfig::default();
    let pool = sqlx_adapter::SqlxAdapter::connect(&config, "sqlite::memory:", Duration::from_secs(2)).await.unwrap();
    sqlx::query("CREATE TABLE rollback_check (v TEXT)").execute(&pool).await.unwrap();
    let result = sqlx_adapter::SqlxAdapter::transaction(&pool, |tx| Box::pin(async move {
        sqlx::query("INSERT INTO rollback_check VALUES (?)").bind("temporary").execute(&mut **tx).await.map_err(|e| lust::stdlib::net::NetError::Io(e.to_string()))?;
        Err(lust::stdlib::net::NetError::Io("force rollback".into()))
    })).await;
    assert!(result.is_err());
    let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM rollback_check").fetch_one(&pool).await.unwrap();
    assert_eq!(count, 0);
}

#[cfg(all(feature = "net-services", feature = "svc-sqlx"))]
#[tokio::test]
async fn sqlx_connect_timeout_is_reported() {
    let config = LustConfig::default();
    let result = sqlx_adapter::SqlxAdapter::connect(&config, "sqlite::memory:", Duration::ZERO).await;
    assert!(matches!(result, Err(lust::stdlib::net::NetError::Timeout(_))));
}

#[cfg(all(feature = "net-services", feature = "svc-reqwest"))]
#[tokio::test]
async fn reqwest_reports_http_error_status() {
    use std::{io::{Read, Write}, net::TcpListener, thread};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0; 1024]; let _ = stream.read(&mut buf);
        stream.write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 3\r\nConnection: close\r\n\r\nbad").unwrap();
    });
    let (status, body) = reqwest_adapter::ReqwestAdapter::get(&LustConfig::default(), &format!("http://{addr}/"), Duration::from_secs(2), &CancelToken::new()).await.unwrap();
    assert_eq!(status, 500); assert_eq!(body, b"bad"); server.join().unwrap();
}

#[cfg(all(feature = "net-services", feature = "svc-reqwest"))]
#[tokio::test]
async fn reqwest_in_flight_cancellation_is_reported() {
    use std::{io::Read, net::TcpListener, thread};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0; 1024]; let _ = stream.read(&mut buf);
        thread::sleep(Duration::from_millis(250));
    });
    let cancel = CancelToken::new();
    let trigger = cancel.clone();
    tokio::spawn(async move { tokio::time::sleep(Duration::from_millis(50)).await; trigger.cancel(); });
    let result = reqwest_adapter::ReqwestAdapter::get(&LustConfig::default(), &format!("http://{addr}/"), Duration::from_secs(2), &cancel).await;
    assert!(matches!(result, Err(lust::stdlib::net::NetError::Cancelled(_))));
    server.join().unwrap();
}

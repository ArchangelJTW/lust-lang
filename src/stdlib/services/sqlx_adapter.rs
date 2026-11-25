//! Optional SQLx SQLite adapter with bound parameters and rollback-on-error transactions.
#![cfg(feature = "svc-sqlx")]

use lust::{config::LustConfig, stdlib::net::{check_net_capability, NetError}};
use sqlx::{sqlite::SqlitePoolOptions, SqlitePool};
use std::time::Duration;

pub struct SqlxAdapter;

impl SqlxAdapter {
    pub async fn connect(config: &LustConfig, url: &str, timeout: Duration) -> Result<SqlitePool, NetError> {
        check_net_capability(config, "sqlite")?;
        if !config.capability_mask().allow_fs && !url.contains(":memory:") {
            return Err(NetError::CapabilityDenied("file-backed sqlite requires allow_fs".into()));
        }
        tokio::time::timeout(timeout.min(Duration::from_secs(30)), SqlitePoolOptions::new().max_connections(1).connect(url)).await
            .map_err(|_| NetError::Timeout("SQLite connect timed out".into()))?
            .map_err(|e| NetError::Io(e.to_string()))
    }

    pub async fn execute(pool: &SqlitePool, sql: &'static str, value: &str) -> Result<u64, NetError> {
        sqlx::query(sql).bind(value.to_owned()).execute(pool).await.map(|r| r.rows_affected()).map_err(|e| NetError::Io(e.to_string()))
    }

    pub async fn transaction<F>(pool: &SqlitePool, operation: F) -> Result<(), NetError>
    where
        F: for<'c> FnOnce(
            &'c mut sqlx::Transaction<'static, sqlx::Sqlite>,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), NetError>> + Send + 'c>>,
    {
        let mut tx = pool.begin().await.map_err(|e| NetError::Io(e.to_string()))?;
        match operation(&mut tx).await {
            Ok(()) => tx.commit().await.map_err(|e| NetError::Io(e.to_string())),
            Err(e) => { let _ = tx.rollback().await; Err(e) }
        }
    }
}

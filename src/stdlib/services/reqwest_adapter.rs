//! Optional reqwest based HTTP client. Callers supply a capability checked config.
#![cfg(feature = "svc-reqwest")]

use std::time::Duration;

use lust::{config::LustConfig, stdlib::net::{check_net_capability, CancelToken, NetError}};

/// Bounded HTTP operations backed by reqwest. Capability denial precedes client creation.
pub struct ReqwestAdapter;

impl ReqwestAdapter {
    pub async fn get(config: &LustConfig, url: &str, timeout: Duration, cancel: &CancelToken) -> Result<(u16, Vec<u8>), NetError> {
        check_net_capability(config, "http")?;
        if cancel.is_cancelled() { return Err(NetError::Cancelled("HTTP request cancelled".into())); }
        let timeout = timeout.min(Duration::from_secs(30));
        let client = reqwest::Client::builder().timeout(timeout).build().map_err(|e| NetError::Io(e.to_string()))?;
        tokio::select! {
            _ = async { while !cancel.is_cancelled() { tokio::time::sleep(Duration::from_millis(10)).await; } } => Err(NetError::Cancelled("HTTP request cancelled".into())),
            result = client.get(url).send() => {
                let response = result.map_err(|e| if e.is_timeout() { NetError::Timeout(e.to_string()) } else { NetError::Io(e.to_string()) })?;
                let status = response.status().as_u16();
                let body = response.bytes().await.map_err(|e| NetError::Io(e.to_string()))?.to_vec();
                Ok((status, body))
            }
        }
    }
}

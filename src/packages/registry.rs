use crate::packages::{archive::PackageArchive, manifest::PackageManifest};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

pub const DEFAULT_BASE_URL: &str = "https://lust-lang.dev/";

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("failed to execute curl command: {0}")]
    Spawn(#[from] io::Error),

    #[error("registry request failed with status {status}: {stderr}")]
    CommandFailed { status: i32, stderr: String },

    #[error("registry responded with status {status}: {message}")]
    Api {
        status: u16,
        code: Option<String>,
        message: String,
    },

    #[error("failed to parse registry response: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct RegistryClient {
    base_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PublishResponse {
    pub package: String,
    pub version: String,
    pub artifact_sha256: String,
    pub download_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PackageSearchResponse {
    pub total: u64,
    pub page: u32,
    pub per_page: u32,
    pub sort: String,
    pub packages: Vec<PackageSummary>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PackageSummary {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub categories: Vec<String>,
    pub downloads: Option<u64>,
    pub updated_at: Option<String>,
    pub latest_version: Option<PackageVersionInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PackageDetails {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub categories: Vec<String>,
    pub downloads: Option<u64>,
    pub updated_at: Option<String>,
    pub latest_version: Option<PackageVersionInfo>,
    #[serde(default)]
    pub versions: Vec<PackageVersionInfo>,
    #[serde(default)]
    pub readme_html: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PackageVersionInfo {
    pub version: String,
    pub published_at: Option<String>,
}

#[derive(Debug, Default)]
pub struct SearchParameters {
    pub q: Option<String>,
    pub keyword: Option<String>,
    pub category: Option<String>,
    pub sort: Option<String>,
    pub page: Option<u32>,
    pub per_page: Option<u32>,
}

#[derive(Debug)]
pub struct DownloadedArchive {
    path: PathBuf,
}

impl DownloadedArchive {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn into_path(self) -> PathBuf {
        let path = self.path.clone();
        std::mem::forget(self);
        path
    }
}

impl Drop for DownloadedArchive {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

impl RegistryClient {
    pub fn new(base: &str) -> Result<Self, RegistryError> {
        let base_url = if base.ends_with('/') {
            base.to_string()
        } else {
            format!("{base}/")
        };
        Ok(Self { base_url })
    }

    pub fn publish(
        &self,
        manifest: &PackageManifest,
        token: &str,
        archive: &PackageArchive,
        readme: Option<&str>,
    ) -> Result<PublishResponse, RegistryError> {
        let metadata_json = serde_json::to_string(&manifest.metadata_payload())?;
        let metadata_file = TempFile::write("lust-metadata", "json", metadata_json.as_bytes())?;
        let readme_file = if let Some(content) = readme {
            if content.trim().is_empty() {
                None
            } else {
                Some(TempFile::write("lust-readme", "md", content.as_bytes())?)
            }
        } else {
            None
        };

        let response_file = TempFile::empty("lust-response", "json")?;
        let mut command = self.base_curl_command();
        command.arg("--output").arg(response_file.path());
        command.arg("--write-out").arg("%{http_code}");
        command.arg("-X").arg("POST");
        command.arg(self.join("api/publish"));
        command
            .arg("-H")
            .arg(format!("Authorization: Bearer {}", token));
        command.arg("-F").arg(format!(
            "metadata=@{};type=application/json",
            metadata_file.path().display()
        ));
        command.arg("-F").arg(format!(
            "artifact=@{};type=application/gzip",
            archive.path().display()
        ));
        if let Some(readme_temp) = &readme_file {
            command
                .arg("-F")
                .arg(format!("readme=@{}", readme_temp.path().display()));
        }

        let output = command.output()?;
        let status_code = parse_status_code(&output)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(RegistryError::CommandFailed {
                status: output.status.code().unwrap_or(-1),
                stderr,
            });
        }
        let body = response_file.read_to_string()?;
        if status_code >= 400 {
            return Err(parse_api_error(status_code, &body));
        }
        let response = serde_json::from_str(&body)?;
        Ok(response)
    }

    pub fn package_details(&self, name: &str) -> Result<PackageDetails, RegistryError> {
        let url = self.join(&format!("api/packages/{}", encode_segment(name)));
        let (status, body) = self.get_json(&url)?;
        if status >= 400 {
            return Err(parse_api_error(status, &body));
        }
        Ok(serde_json::from_str(&body)?)
    }

    pub fn search_packages(
        &self,
        params: &SearchParameters,
    ) -> Result<PackageSearchResponse, RegistryError> {
        let mut url = self.join("api/packages");
        let mut query: Vec<(String, String)> = Vec::new();
        if let Some(q) = &params.q {
            query.push(("q".to_string(), encode_query_value(q)));
        }
        if let Some(keyword) = &params.keyword {
            query.push(("keyword".to_string(), encode_query_value(keyword)));
        }
        if let Some(category) = &params.category {
            query.push(("category".to_string(), encode_query_value(category)));
        }
        if let Some(sort) = &params.sort {
            query.push(("sort".to_string(), encode_query_value(sort)));
        }
        if let Some(page) = params.page {
            query.push(("page".to_string(), page.to_string()));
        }
        if let Some(per_page) = params.per_page {
            query.push(("per_page".to_string(), per_page.to_string()));
        }
        if !query.is_empty() {
            let query_string = query
                .into_iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join("&");
            url.push('?');
            url.push_str(&query_string);
        }
        let (status, body) = self.get_json(&url)?;
        if status >= 400 {
            return Err(parse_api_error(status, &body));
        }
        Ok(serde_json::from_str(&body)?)
    }

    pub fn download_package(
        &self,
        name: &str,
        version: &str,
    ) -> Result<DownloadedArchive, RegistryError> {
        let url = self.join(&format!(
            "api/packages/{}/{}/download",
            encode_segment(name),
            encode_segment(version)
        ));
        let output_path = temp_path("lust-download", "tar.gz");
        let mut command = self.base_curl_command();
        command.arg("--output").arg(&output_path);
        command.arg("--write-out").arg("%{http_code}");
        command.arg(url);
        let output = command.output()?;
        let status_code = parse_status_code(&output)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(RegistryError::CommandFailed {
                status: output.status.code().unwrap_or(-1),
                stderr,
            });
        }
        if status_code >= 400 {
            let body = fs::read_to_string(&output_path).unwrap_or_default();
            fs::remove_file(&output_path).ok();
            return Err(parse_api_error(status_code, &body));
        }
        Ok(DownloadedArchive { path: output_path })
    }

    fn get_json(&self, url: &str) -> Result<(u16, String), RegistryError> {
        let response_file = TempFile::empty("lust-response", "json")?;
        let mut command = self.base_curl_command();
        command.arg("--output").arg(response_file.path());
        command.arg("--write-out").arg("%{http_code}");
        command.arg(url);
        let output = command.output()?;
        let status_code = parse_status_code(&output)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(RegistryError::CommandFailed {
                status: output.status.code().unwrap_or(-1),
                stderr,
            });
        }
        let body = response_file.read_to_string()?;
        Ok((status_code, body))
    }

    fn base_curl_command(&self) -> Command {
        let mut command = Command::new(resolve_curl_command());
        command.arg("-sS");
        command
    }

    fn join(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

fn resolve_curl_command() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "curl.exe"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "curl"
    }
}

fn encode_segment(input: &str) -> String {
    const UNRESERVED: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    const HEX: &[u8] = b"0123456789ABCDEF";
    let mut encoded = String::new();
    for byte in input.bytes() {
        if UNRESERVED.contains(&byte) {
            encoded.push(byte as char);
        } else {
            encoded.push('%');
            encoded.push(HEX[(byte >> 4) as usize] as char);
            encoded.push(HEX[(byte & 0xF) as usize] as char);
        }
    }
    encoded
}

fn encode_query_value(input: &str) -> String {
    encode_segment(input)
}

fn parse_status_code(output: &Output) -> Result<u16, RegistryError> {
    let status_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
    status_str
        .parse::<u16>()
        .map_err(|_| RegistryError::CommandFailed {
            status: output.status.code().unwrap_or(-1),
            stderr: format!("invalid status code from curl: {}", status_str),
        })
}

fn parse_api_error(status: u16, body: &str) -> RegistryError {
    if let Ok(json) = serde_json::from_str::<JsonValue>(body) {
        let code = json.get("code").and_then(|v| v.as_str()).map(String::from);
        let message = json
            .get("message")
            .and_then(|v| v.as_str())
            .map(String::from)
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| body.trim().to_string());
        RegistryError::Api {
            status,
            code,
            message,
        }
    } else {
        RegistryError::Api {
            status,
            code: None,
            message: body.trim().to_string(),
        }
    }
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn write(prefix: &str, ext: &str, contents: &[u8]) -> io::Result<Self> {
        let path = temp_path(prefix, ext);
        fs::write(&path, contents)?;
        Ok(Self { path })
    }

    fn empty(prefix: &str, ext: &str) -> io::Result<Self> {
        let path = temp_path(prefix, ext);
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn read_to_string(&self) -> io::Result<String> {
        fs::read_to_string(&self.path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn temp_path(prefix: &str, ext: &str) -> PathBuf {
    let mut path = env::temp_dir();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();
    let pid = std::process::id();
    path.push(format!("{prefix}-{pid}-{timestamp}.{ext}"));
    path
}

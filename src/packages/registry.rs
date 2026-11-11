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
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

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

pub fn compute_sha256(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = Vec::with_capacity(data.len() + 64);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, part) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([part[0], part[1], part[2], part[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut h_val = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h_val
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h_val = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(h_val);
    }

    use std::fmt::Write;
    let mut out = String::with_capacity(64);
    for val in &h {
        let _ = write!(&mut out, "{:08x}", val);
    }
    out
}

pub fn verify_sha256(data: &[u8], expected_hex: &str) -> bool {
    let computed = compute_sha256(data);
    computed.eq_ignore_ascii_case(expected_hex.trim())
}

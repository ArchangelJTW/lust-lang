use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

const SKIP_PATTERNS: &[&str] = &[
    "target",
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "node_modules",
    "__pycache__",
    ".DS_Store",
];

#[derive(Debug, Error)]
pub enum ArchiveError {
    #[error("package root {0} does not exist")]
    RootMissing(PathBuf),

    #[error("failed to spawn tar command: {0}")]
    Spawn(#[from] io::Error),

    #[error("tar command failed with status {status}: {stderr}")]
    CommandFailed { status: i32, stderr: String },
}

#[derive(Debug)]
pub struct PackageArchive {
    path: PathBuf,
}

impl PackageArchive {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn into_path(self) -> PathBuf {
        let path = self.path.clone();
        std::mem::forget(self);
        path
    }
}

impl Drop for PackageArchive {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub fn build_package_archive(root: &Path) -> Result<PackageArchive, ArchiveError> {
    if !root.exists() {
        return Err(ArchiveError::RootMissing(root.to_path_buf()));
    }
    let output_path = temp_archive_path();
    let mut command = Command::new(resolve_tar_command());
    command.arg("-czf");
    command.arg(&output_path);
    for pattern in SKIP_PATTERNS {
        command.arg(format!("--exclude={pattern}"));
    }
    command.arg("-C");
    command.arg(root);
    command.arg(".");
    let output = command.output()?;
    ensure_success(output)?;
    Ok(PackageArchive { path: output_path })
}

fn resolve_tar_command() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "tar.exe"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "tar"
    }
}

fn temp_archive_path() -> PathBuf {
    let mut path = env::temp_dir();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();
    let pid = std::process::id();
    path.push(format!("lust-package-{pid}-{timestamp}.tar.gz"));
    path
}

fn ensure_success(output: Output) -> Result<(), ArchiveError> {
    if output.status.success() {
        Ok(())
    } else {
        let code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(ArchiveError::CommandFailed {
            status: code,
            stderr,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;
    use tempfile::tempdir;

    fn list_archive_contents(path: &Path) -> Vec<String> {
        let output = Command::new(resolve_tar_command())
            .arg("-tzf")
            .arg(path)
            .output()
            .expect("tar -tzf");
        assert!(output.status.success(), "tar -tzf failed");
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|line| {
                let trimmed = line.trim();
                trimmed.strip_prefix("./").unwrap_or(trimmed).to_string()
            })
            .collect()
    }

    #[test]
    fn archive_skips_target_directory() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("target/cache")).unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.lust"), "content").unwrap();
        fs::write(root.join("target/cache.bin"), "ignore").unwrap();

        let archive = build_package_archive(root).unwrap();
        let entries = list_archive_contents(archive.path());
        assert!(entries.iter().any(|entry| entry == "src/lib.lust"));
        assert!(!entries.iter().any(|entry| entry.starts_with("target/")));
    }
}

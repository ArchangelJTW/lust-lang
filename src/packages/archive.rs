use std::{
    env,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use toml::Value;

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

    #[error("failed to stage package contents: {source}")]
    StageIo {
        #[source]
        source: io::Error,
    },

    #[error("failed to sanitize manifest {path}: {source}")]
    SanitizeParse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("failed to sanitize manifest {path}: {source}")]
    SanitizeSerialize {
        path: PathBuf,
        #[source]
        source: toml::ser::Error,
    },

    #[error("failed to sanitize manifest {path}: {source}")]
    SanitizeIo {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
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
    let staging_dir = create_staging_dir().map_err(|source| ArchiveError::StageIo { source })?;
    let archive_result = (|| -> Result<PackageArchive, ArchiveError> {
        copy_project(root, &staging_dir)?;
        sanitize_manifests(&staging_dir)?;
        let output_path = temp_archive_path();
        let mut command = Command::new(resolve_tar_command());
        command.arg("-czf");
        command.arg(&output_path);
        for pattern in SKIP_PATTERNS {
            command.arg(format!("--exclude={pattern}"));
        }
        command.arg("-C");
        command.arg(&staging_dir);
        command.arg(".");
        let output = command.output()?;
        ensure_success(output)?;
        Ok(PackageArchive { path: output_path })
    })();
    let cleanup_result = fs::remove_dir_all(&staging_dir);
    match (archive_result, cleanup_result) {
        (Ok(archive), Ok(())) => Ok(archive),
        (Ok(_), Err(err)) => Err(ArchiveError::StageIo { source: err }),
        (Err(err), _) => Err(err),
    }
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

fn create_staging_dir() -> io::Result<PathBuf> {
    let mut path = env::temp_dir();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();
    let pid = std::process::id();
    path.push(format!("lust-package-staging-{pid}-{timestamp}"));
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn copy_project(src: &Path, dst: &Path) -> Result<(), ArchiveError> {
    copy_recursive(src, dst)?;
    Ok(())
}

fn copy_recursive(src: &Path, dst: &Path) -> Result<(), ArchiveError> {
    fs::create_dir_all(dst).map_err(|source| ArchiveError::StageIo { source })?;
    for entry in fs::read_dir(src).map_err(|source| ArchiveError::StageIo { source })? {
        let entry = entry.map_err(|source| ArchiveError::StageIo { source })?;
        let file_name = entry.file_name();
        if should_skip(&file_name) {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&file_name);
        let file_type = entry
            .file_type()
            .map_err(|source| ArchiveError::StageIo { source })?;
        if file_type.is_dir() {
            copy_recursive(&src_path, &dst_path)?;
        } else if file_type.is_file() {
            if let Err(source) = fs::copy(&src_path, &dst_path) {
                return Err(ArchiveError::StageIo { source });
            }
        } else if file_type.is_symlink() {
            // replicate symlink as copy of target contents for portability
            let target =
                fs::read_link(&src_path).map_err(|source| ArchiveError::StageIo { source })?;
            let resolved = if target.is_absolute() {
                target
            } else {
                src_path.parent().unwrap_or(src).join(target)
            };
            if let Err(source) = fs::copy(&resolved, &dst_path) {
                return Err(ArchiveError::StageIo { source });
            }
        }
    }
    Ok(())
}

fn should_skip(name: &OsStr) -> bool {
    let name = name.to_string_lossy();
    SKIP_PATTERNS.iter().any(|pattern| name == *pattern)
}

fn sanitize_manifests(root: &Path) -> Result<(), ArchiveError> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).map_err(|source| ArchiveError::StageIo { source })? {
            let entry = entry.map_err(|source| ArchiveError::StageIo { source })?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().and_then(|n| n.to_str()) == Some("Cargo.toml") {
                sanitize_manifest(&path)?;
            }
        }
    }
    Ok(())
}

fn sanitize_manifest(path: &Path) -> Result<(), ArchiveError> {
    let original = fs::read_to_string(path).map_err(|source| ArchiveError::SanitizeIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut value: Value =
        toml::from_str(&original).map_err(|source| ArchiveError::SanitizeParse {
            path: path.to_path_buf(),
            source,
        })?;
    let mut changed = false;
    if let Some(table) = value.as_table_mut() {
        sanitize_dependency_tables(table, &mut changed);
    }
    if changed {
        let serialized =
            toml::to_string_pretty(&value).map_err(|source| ArchiveError::SanitizeSerialize {
                path: path.to_path_buf(),
                source,
            })?;
        fs::write(path, serialized).map_err(|source| ArchiveError::SanitizeIo {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(())
}

fn sanitize_dependency_tables(table: &mut toml::value::Table, changed: &mut bool) {
    for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(value) = table.get_mut(key) {
            if let Value::Table(dep_table) = value {
                sanitize_dependency_table(dep_table, changed);
            }
        }
    }
    for (_, value) in table.iter_mut() {
        if let Value::Table(sub) = value {
            sanitize_dependency_tables(sub, changed);
        }
    }
}

fn sanitize_dependency_table(table: &mut toml::value::Table, changed: &mut bool) {
    for (_, value) in table.iter_mut() {
        if let Value::Table(spec) = value {
            if sanitize_spec_table(spec) {
                *changed = true;
            }
        }
    }
}

fn sanitize_spec_table(spec: &mut toml::value::Table) -> bool {
    let has_version = spec.contains_key("version");
    if has_version {
        spec.remove("path").is_some()
    } else {
        false
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

    #[test]
    fn archive_strips_path_dependencies() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn foo() {}\n").unwrap();
        fs::write(
            root.join("Cargo.toml"),
            r#"
                [package]
                name = "path-test"
                version = "0.1.0"
                edition = "2021"

                [dependencies]
                lust = { version = "1.2.3", path = "../lust" }
            "#,
        )
        .unwrap();

        let archive = build_package_archive(root).unwrap();
        let unpack = tempdir().unwrap();
        let status = Command::new(resolve_tar_command())
            .arg("-xzf")
            .arg(archive.path())
            .arg("-C")
            .arg(unpack.path())
            .status()
            .expect("tar -xzf");
        assert!(status.success(), "tar extraction failed");

        let manifest_path = unpack.path().join("Cargo.toml");
        assert!(manifest_path.exists());
        let contents = fs::read_to_string(&manifest_path).unwrap();
        let parsed: Value = toml::from_str(&contents).unwrap();
        let deps = parsed
            .get("dependencies")
            .and_then(Value::as_table)
            .expect("dependencies table missing");
        let lust_entry = deps
            .get("lust")
            .and_then(Value::as_table)
            .expect("lust dependency missing");
        assert!(lust_entry.get("version").is_some());
        assert!(
            lust_entry.get("path").is_none(),
            "path key should be stripped"
        );
    }
}

use dirs::home_dir;
use std::{
    fs,
    io::{self, Read, Write},
    path::PathBuf,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CredentialsError {
    #[error("unable to determine user home directory")]
    HomeDirUnavailable,

    #[error("failed to access credentials at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}

#[derive(Debug, Clone)]
pub struct Credentials {
    token: String,
}

impl Credentials {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }

    pub fn token(&self) -> &str {
        &self.token
    }
}

pub fn credentials_file() -> Result<PathBuf, CredentialsError> {
    let home = home_dir().ok_or(CredentialsError::HomeDirUnavailable)?;
    let dir = home.join(".lust");
    Ok(dir.join("credentials"))
}

pub fn load_credentials() -> Result<Option<Credentials>, CredentialsError> {
    let path = credentials_file()?;
    match fs::File::open(&path) {
        Ok(mut file) => {
            let mut buf = String::new();
            file.read_to_string(&mut buf)
                .map_err(|source| CredentialsError::Io {
                    path: path.clone(),
                    source,
                })?;
            let token = buf.trim().to_string();
            if token.is_empty() {
                Ok(None)
            } else {
                Ok(Some(Credentials::new(token)))
            }
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(CredentialsError::Io { path, source }),
    }
}

pub fn save_credentials(token: &str) -> Result<(), CredentialsError> {
    let path = credentials_file()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| CredentialsError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut file = fs::File::create(&path).map_err(|source| CredentialsError::Io {
        path: path.clone(),
        source,
    })?;
    file.write_all(token.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .map_err(|source| CredentialsError::Io { path, source })
}

pub fn clear_credentials() -> Result<(), CredentialsError> {
    let path = credentials_file()?;
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(CredentialsError::Io { path, source }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use tempfile::tempdir;

    #[test]
    fn save_and_load_credentials() {
        let dir = tempdir().unwrap();
        let original_home = env::var("HOME").ok();
        let original_userprofile = env::var("USERPROFILE").ok();
        env::set_var("HOME", dir.path());
        env::set_var("USERPROFILE", dir.path());

        save_credentials("secret-token").unwrap();
        let creds = load_credentials().unwrap().unwrap();
        assert_eq!(creds.token(), "secret-token");

        let path = credentials_file().unwrap();
        assert!(path.exists());

        env::remove_var("HOME");
        env::remove_var("USERPROFILE");
        if let Some(home) = original_home {
            env::set_var("HOME", home);
        }
        if let Some(userprofile) = original_userprofile {
            env::set_var("USERPROFILE", userprofile);
        }
    }
}

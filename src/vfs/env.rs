use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// VirtualEnv: Isolated environment variable table per VM instance.
/// Virtualizes `os.getenv` and `os.setenv` to prevent host environment pollution and leaks.
#[derive(Debug, Clone)]
pub struct VirtualEnv {
    vars: Arc<RwLock<HashMap<String, String>>>,
    isolated: bool,
}

impl Default for VirtualEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualEnv {
    pub fn new() -> Self {
        Self {
            vars: Arc::new(RwLock::new(HashMap::new())),
            isolated: false,
        }
    }

    pub fn isolated() -> Self {
        Self {
            vars: Arc::new(RwLock::new(HashMap::new())),
            isolated: true,
        }
    }

    pub fn with_vars(vars: HashMap<String, String>) -> Self {
        Self {
            vars: Arc::new(RwLock::new(vars)),
            isolated: true,
        }
    }

    pub fn is_isolated(&self) -> bool {
        self.isolated || crate::sim::is_simulated_mode()
    }

    pub fn get(&self, key: &str) -> Option<String> {
        if let Some(val) = self.vars.read().unwrap().get(key) {
            return Some(val.clone());
        }
        if !self.is_isolated() {
            std::env::var(key).ok()
        } else {
            None
        }
    }

    pub fn set(&self, key: impl Into<String>, val: impl Into<String>) {
        self.vars.write().unwrap().insert(key.into(), val.into());
    }

    pub fn remove(&self, key: &str) -> Option<String> {
        self.vars.write().unwrap().remove(key)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.vars.read().unwrap().contains_key(key) || (!self.is_isolated() && std::env::var(key).is_ok())
    }

    pub fn clone_vars(&self) -> HashMap<String, String> {
        self.vars.read().unwrap().clone()
    }
}

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::effect_abi::{Cause, Exit, InterruptReason};
use crate::vm::VM;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvError {
    NotFound(String),
    InvalidEncoding(String),
    CapabilityDenied(String),
}

impl core::fmt::Display for EnvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EnvError::NotFound(s) => write!(f, "NotFound: {}", s),
            EnvError::InvalidEncoding(s) => write!(f, "InvalidEncoding: {}", s),
            EnvError::CapabilityDenied(s) => write!(f, "CapabilityDenied: {}", s),
        }
    }
}

impl std::error::Error for EnvError {}

pub struct EnvService;

impl EnvService {
    pub fn check_capability(vm: &VM) -> Result<(), EnvError> {
        let caps = vm.config().capability_mask();
        if !caps.allow_os {
            return Err(EnvError::CapabilityDenied(
                "OS/Env access denied: capability 'allow_os' disabled".to_string(),
            ));
        }
        if !vm.config().is_std_module_allowed("os") {
            return Err(EnvError::CapabilityDenied(
                "OS/Env access denied: module 'os' is not allowed".to_string(),
            ));
        }
        Ok(())
    }

    pub fn var(vm: &mut VM, key: &str) -> Exit<String, EnvError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(5).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        match vm.env.get(key) {
            Some(val) => {
                if !vm.try_charge_memory_bytes(val.len()) {
                    return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfMemory));
                }
                Exit::Success(val)
            }
            None => Exit::Failure(Cause::Fail(EnvError::NotFound(format!(
                "Environment variable '{}' not found",
                key
            )))),
        }
    }

    pub fn set_var(vm: &mut VM, key: &str, val: &str) -> Exit<(), EnvError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(5 + (val.len() as u64 / 64)).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        vm.env.set(key, val);
        Exit::Success(())
    }

    pub fn args(vm: &mut VM) -> Exit<Vec<String>, EnvError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(5).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        Exit::Success(Vec::new())
    }

    pub fn now_millis(vm: &mut VM) -> Exit<u64, EnvError> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(5).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        let nanos = vm.clock.now_epoch_nanos();
        Exit::Success(nanos / 1_000_000)
    }
}

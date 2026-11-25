use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use rand::RngExt;

use crate::effect_abi::{Cause, Exit, InterruptReason};
use crate::vm::VM;

pub struct RandomService;

impl RandomService {
    pub fn check_capability(vm: &VM) -> Result<(), String> {
        if !vm.config().is_std_module_allowed("random")
            && !vm.config().capability_mask().custom_capabilities.contains("random")
        {
            return Err("CapabilityDenied: module 'random' is not allowed".to_string());
        }
        Ok(())
    }

    pub fn next_int(vm: &mut VM, min: i64, max: i64) -> Exit<i64, String> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if min >= max {
            return Exit::Failure(Cause::Fail(format!(
                "Invalid range: min ({}) >= max ({})",
                min, max
            )));
        }
        if vm.charge_gas(5).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        let val = vm.rng.borrow_mut().random_range(min..max);
        Exit::Success(val)
    }

    pub fn next_float(vm: &mut VM) -> Exit<f64, String> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(5).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        let val = vm.rng.borrow_mut().random::<f64>();
        Exit::Success(val)
    }

    pub fn next_bool(vm: &mut VM) -> Exit<bool, String> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(5).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        let val = vm.rng.borrow_mut().random::<bool>();
        Exit::Success(val)
    }

    pub fn fill_bytes(vm: &mut VM, len: usize) -> Exit<Vec<u8>, String> {
        if let Err(e) = Self::check_capability(vm) {
            return Exit::Failure(Cause::Fail(e));
        }
        if vm.charge_gas(5 + (len as u64 / 16)).is_err() {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfGas));
        }
        if !vm.try_charge_memory_bytes(len) {
            return Exit::Failure(Cause::Interrupted(InterruptReason::OutOfMemory));
        }
        let mut buf = alloc::vec![0u8; len];
        vm.rng.borrow_mut().fill(&mut buf[..]);
        Exit::Success(buf)
    }
}

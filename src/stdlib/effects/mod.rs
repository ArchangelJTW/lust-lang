use alloc::format;
use alloc::string::String;
use crate::effect_abi::{Cause, Exit};

pub mod env;
pub mod fs;
pub mod random;

pub use env::{EnvError, EnvService};
pub use fs::{FileHandle, FileSystemService, IoError};
pub use random::RandomService;

/// Helper to convert Exit into a Value map representation for Lust script interop.
pub fn exit_to_value<E: core::fmt::Display>(exit: impl core::borrow::Borrow<Exit<crate::bytecode::Value, E>>) -> crate::bytecode::Value {
    use crate::bytecode::{Value, ValueKey};
    let key_effect = ValueKey::from(String::from("__effect"));
    let key_kind = ValueKey::from(String::from("__kind"));

    match exit.borrow() {
        Exit::Success(val) => {
            let key_val = ValueKey::from(String::from("__value"));
            let entries = vec![
                (key_effect, Value::Bool(true)),
                (key_kind, Value::string("succeed")),
                (key_val, val.clone()),
            ];
            Value::map(entries.into_iter().collect())
        }
        Exit::Failure(Cause::Fail(err)) => {
            let key_err = ValueKey::from(String::from("__error"));
            let entries = vec![
                (key_effect, Value::Bool(true)),
                (key_kind, Value::string("fail")),
                (key_err, Value::string(err.to_string())),
            ];
            Value::map(entries.into_iter().collect())
        }
        Exit::Failure(Cause::Interrupted(reason)) => {
            let key_reason = ValueKey::from(String::from("__reason"));
            let entries = vec![
                (key_effect, Value::Bool(true)),
                (key_kind, Value::string("interrupt")),
                (key_reason, Value::string(format!("{:?}", reason))),
            ];
            Value::map(entries.into_iter().collect())
        }
        Exit::Failure(Cause::Interrupt) => {
            let key_reason = ValueKey::from(String::from("__reason"));
            let entries = vec![
                (key_effect, Value::Bool(true)),
                (key_kind, Value::string("interrupt")),
                (key_reason, Value::string("Cancelled")),
            ];
            Value::map(entries.into_iter().collect())
        }
        Exit::Failure(Cause::Die(defect)) => {
            let key_defect = ValueKey::from(String::from("__defect"));
            let entries = vec![
                (key_effect, Value::Bool(true)),
                (key_kind, Value::string("die")),
                (key_defect, Value::string(defect.clone())),
            ];
            Value::map(entries.into_iter().collect())
        }
    }
}

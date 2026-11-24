//! Real script-level view/borrow runtime bridge for `EmbeddedProgram`.
//!
//! Installs the runtime implementations for compiler borrow/view intrinsics:
//! - `__lust_borrow(owner)` / `view.borrow(owner)`
//! - `__lust_borrow_mut(owner)` / `view.borrow_mut(owner)`
//! - `__lust_view_read(v)` / `view.read(v)`
//! - `__lust_materialize(v)` / `view.materialize(v)` (true deep independent clone!)
//!
//! Static safety (escape rejection, closure capture rejection, suspension/task rejection,
//! backing mutation conflicts) is enforced BEFORE execution by `src/typechecker/lifetime.rs`.
//!
//! 17C/17D Integration Proposal (`NG-W17-9`):
//! - Wire `install_view_runtime` into default VM / `EmbeddedProgram` startup (`src/vm/stdlib.rs`).

use std::rc::Rc;

use crate::bytecode::value::{EnumObject, StructObject};
use crate::bytecode::{LustMap, NativeCallResult, Value, ValueKey, native_fn};
use crate::embed::{
    EmbeddedBuilder, EmbeddedProgram, FunctionBuilder, function_param, type_unknown,
};

/// Declares the compiler borrow & view intrinsics (`__lust_borrow`, `__lust_borrow_mut`,
/// `__lust_view_read`, `__lust_materialize`, and namespace `view`) in `builder`'s extern
/// registry so the bytecode compiler emits global lookups for them.
pub fn declare_view_externs(mut builder: EmbeddedBuilder) -> EmbeddedBuilder {
    for name in [
        "__lust_borrow",
        "__lust_borrow_mut",
        "__lust_view_read",
        "__lust_materialize",
    ] {
        let def = FunctionBuilder::new(name)
            .param(function_param("value", type_unknown()))
            .return_type(type_unknown())
            .finish();
        builder = builder.declare_function(def);
    }
    builder
        .extern_registry_mut()
        .add_const("view", type_unknown());
    builder
}

/// Deeply clones a Lust `Value` into brand-new independent backing storage (`Rc`/`RefCell`),
/// guaranteeing that post-materialization mutation of the source container or nested structs
/// cannot observe or alias the materialized copy.
pub fn deep_clone_value(val: &Value) -> Result<Value, String> {
    deep_clone_value_bounded(val, 64)
}

fn deep_clone_value_bounded(val: &Value, depth_remaining: usize) -> Result<Value, String> {
    if depth_remaining == 0 {
        return Err("view.materialize exceeded maximum nesting depth (possible cycle)".to_string());
    }
    match val {
        Value::Nil => Ok(Value::Nil),
        Value::Bool(b) => Ok(Value::Bool(*b)),
        Value::Int(i) => Ok(Value::Int(*i)),
        Value::Float(f) => Ok(Value::Float(*f)),
        Value::String(s) => Ok(Value::String(Rc::new((**s).clone()))),
        Value::Array(arr) => {
            let borrowed = arr.borrow();
            let mut cloned = Vec::with_capacity(borrowed.len());
            for item in borrowed.iter() {
                cloned.push(deep_clone_value_bounded(item, depth_remaining - 1)?);
            }
            Ok(Value::array(cloned))
        }
        Value::Tuple(items) => {
            let mut cloned = Vec::with_capacity(items.len());
            for item in items.iter() {
                cloned.push(deep_clone_value_bounded(item, depth_remaining - 1)?);
            }
            Ok(Value::tuple(cloned))
        }
        Value::Map(map) => {
            let borrowed = map.borrow();
            let mut cloned = LustMap::default();
            for (k, v) in borrowed.iter() {
                cloned.insert(k.clone(), deep_clone_value_bounded(v, depth_remaining - 1)?);
            }
            Ok(Value::map(cloned))
        }
        Value::Struct(obj) => {
            let fields = obj.fields.borrow();
            let mut cloned_fields = Vec::with_capacity(fields.len());
            for (idx, field_val) in fields.iter().enumerate() {
                let materialized = obj.layout.materialize_field_value(idx, field_val.clone());
                cloned_fields.push(deep_clone_value_bounded(&materialized, depth_remaining - 1)?);
            }
            Ok(Value::Struct(StructObject::new(
                obj.name.clone(),
                Rc::clone(&obj.layout),
                cloned_fields,
            )))
        }
        Value::Enum(obj) => {
            let cloned_values = match &obj.values {
                Some(vals) => {
                    let mut out = Vec::with_capacity(vals.len());
                    for v in vals {
                        out.push(deep_clone_value_bounded(v, depth_remaining - 1)?);
                    }
                    Some(out)
                }
                None => None,
            };
            Ok(Value::Enum(EnumObject::new(
                obj.enum_name.clone(),
                obj.variant.clone(),
                cloned_values,
            )))
        }
        other => Err(format!(
            "view.materialize does not support non-materializable runtime value {:?}",
            other
        )),
    }
}

fn single_arg<'a>(name: &str, args: &'a [Value]) -> Result<&'a Value, String> {
    if args.len() != 1 {
        return Err(format!("'{}' expects exactly 1 argument, got {}", name, args.len()));
    }
    Ok(&args[0])
}

/// Registers the canonical script borrow & view intrinsics on `program`.
pub fn install_view_runtime(program: &mut EmbeddedProgram) {
    let borrow_impl = |args: &[Value]| -> Result<NativeCallResult, String> {
        let arg = single_arg("__lust_borrow", args)?;
        Ok(NativeCallResult::Return(arg.clone()))
    };
    let borrow_mut_impl = |args: &[Value]| -> Result<NativeCallResult, String> {
        let arg = single_arg("__lust_borrow_mut", args)?;
        Ok(NativeCallResult::Return(arg.clone()))
    };
    let read_impl = |args: &[Value]| -> Result<NativeCallResult, String> {
        let arg = single_arg("view.read", args)?;
        Ok(NativeCallResult::Return(arg.clone()))
    };
    let materialize_impl = |args: &[Value]| -> Result<NativeCallResult, String> {
        let arg = single_arg("view.materialize", args)?;
        Ok(NativeCallResult::Return(deep_clone_value(arg)?))
    };

    program.register_native_fn("__lust_borrow", borrow_impl);
    program.register_native_fn("__lust_borrow_mut", borrow_mut_impl);
    program.register_native_fn("__lust_view_read", read_impl);
    program.register_native_fn("__lust_materialize", materialize_impl);

    let mut view_ns = LustMap::default();
    view_ns.insert(
        ValueKey::string("borrow"),
        Value::NativeFunction(native_fn(borrow_impl)),
    );
    view_ns.insert(
        ValueKey::string("borrow_mut"),
        Value::NativeFunction(native_fn(borrow_mut_impl)),
    );
    view_ns.insert(
        ValueKey::string("read"),
        Value::NativeFunction(native_fn(read_impl)),
    );
    view_ns.insert(
        ValueKey::string("materialize"),
        Value::NativeFunction(native_fn(materialize_impl)),
    );
    program
        .vm_mut()
        .set_global("view", Value::map(view_ns));
}

use lust::{
    ast::{Span, Type, TypeKind},
    bytecode::Value,
    embed::{
        function_param, self_param, struct_field_decl, ExternRegistry, FunctionBuilder,
        ImplBuilder, StructBuilder, StructInstance,
    },
    type_named, VM,
};
use lust_macros::{lust_fn, lust_module};
use std::rc::Rc;

fn ty_int() -> Type {
    Type::new(TypeKind::Int, Span::dummy())
}

fn ty_factor() -> Type {
    type_named("Factor")
}

pub fn register_types(vm: &mut VM) {
    let mut registry = ExternRegistry::new();

    let factor_struct = StructBuilder::new("Factor")
        .field(struct_field_decl("base", ty_int()))
        .field(struct_field_decl("multiplier", ty_int()))
        .finish();
    registry.add_struct(factor_struct);

    let make_signature = FunctionBuilder::new("make_factor")
        .param(function_param("base", ty_int()))
        .param(function_param("multiplier", ty_int()))
        .return_type(ty_factor())
        .finish();
    registry.add_function(make_signature);

    let apply_signature = FunctionBuilder::new("apply")
        .param(self_param(None))
        .param(function_param("value", ty_int()))
        .return_type(ty_int())
        .finish();
    let factor_impl = ImplBuilder::new(type_named("Factor"))
        .method(apply_signature)
        .finish();
    registry.add_impl(factor_impl);

    registry.register_with_vm(vm);
}

#[lust_module(name = "lust_double", abi = "both")]
pub mod lust_double {
    use super::*;

    #[lust_fn(pure)]
    pub fn host_double(value: i64) -> i64 {
        value * 2
    }

    #[lust_fn(name = "nested.host_quadruple", pure)]
    pub fn host_quadruple(value: i64) -> i64 {
        value * 4
    }

    #[lust_fn]
    pub fn make_factor(base: i64, multiplier: i64) -> Result<Value, String> {
        VM::with_current(|vm| {
            vm.instantiate_struct(
                "lust_double.Factor",
                vec![
                    (Rc::new("base".to_string()), Value::Int(base)),
                    (Rc::new("multiplier".to_string()), Value::Int(multiplier)),
                ],
            )
            .map_err(|err| err.to_string())
        })
    }

    #[lust_fn(name = "Factor:apply")]
    pub fn factor_apply(factor: StructInstance, value: i64) -> Result<i64, String> {
        let base: i64 = factor.field("base").map_err(|err| err.to_string())?;
        let multiplier: i64 = factor.field("multiplier").map_err(|err| err.to_string())?;
        Ok(base * value * multiplier)
    }
}

pub use lust_double::*;

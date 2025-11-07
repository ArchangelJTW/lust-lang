use lust::{
    ast::{Span, Type, TypeKind},
    bytecode::{NativeCallResult, Value},
    embed::{
        function_param, self_param, struct_field_decl, ExternRegistry, FunctionBuilder,
        ImplBuilder, StructBuilder, StructInstance,
    },
    type_named, FromLustValue, NativeExport, NativeExportParam, VM,
};
use std::rc::Rc;

fn ty_int() -> Type {
    Type::new(TypeKind::Int, Span::dummy())
}

fn ty_factor() -> Type {
    type_named("Factor")
}

fn register_types(vm: &mut VM) {
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

fn register_functions(vm: &mut VM) -> Result<(), String> {
    register_types(vm);

    let export = NativeExport::new(
        "host_double",
        vec![NativeExportParam::new("value", "int")],
        "int",
    );
    vm.register_exported_native(export, |values: &[Value]| {
        let number = values
            .get(0)
            .and_then(|value| value.as_int())
            .ok_or_else(|| "expected int argument".to_string())?;
        Ok(NativeCallResult::Return(Value::Int(number * 2)))
    });

    let export = NativeExport::new(
        "nested.host_quadruple",
        vec![NativeExportParam::new("value", "int")],
        "int",
    );
    vm.register_exported_native(export, |values: &[Value]| {
        let number = values
            .get(0)
            .and_then(|value| value.as_int())
            .ok_or_else(|| "expected int argument".to_string())?;
        Ok(NativeCallResult::Return(Value::Int(number * 4)))
    });

    let export = NativeExport::new(
        "make_factor",
        vec![
            NativeExportParam::new("base", "int"),
            NativeExportParam::new("multiplier", "int"),
        ],
        "Factor",
    );
    let vm_ptr = vm as *mut VM;
    vm.register_exported_native(export, move |values: &[Value]| {
        let base = values
            .get(0)
            .and_then(|value| value.as_int())
            .ok_or_else(|| "expected base: int".to_string())?;
        let multiplier = values
            .get(1)
            .and_then(|value| value.as_int())
            .ok_or_else(|| "expected multiplier: int".to_string())?;

        let factor_value = unsafe {
            let vm = &mut *vm_ptr;
            vm.instantiate_struct(
                "lust_double.Factor",
                vec![
                    (Rc::new("base".to_string()), Value::Int(base)),
                    (Rc::new("multiplier".to_string()), Value::Int(multiplier)),
                ],
            )
            .map_err(|err| err.to_string())?
        };
        Ok(NativeCallResult::Return(factor_value))
    });

    let export = NativeExport::new(
        "Factor:apply",
        vec![
            NativeExportParam::new("self", "Factor"),
            NativeExportParam::new("value", "int"),
        ],
        "int",
    );
    vm.register_exported_native(export, |values: &[Value]| {
        let factor_value = values
            .get(0)
            .cloned()
            .ok_or_else(|| "expected Factor as first argument".to_string())?;
        let factor = StructInstance::from_value(factor_value).map_err(|err| err.to_string())?;
        let base: i64 = factor.field("base").map_err(|err| err.to_string())?;
        let multiplier: i64 = factor.field("multiplier").map_err(|err| err.to_string())?;
        let input = values
            .get(1)
            .and_then(|value| value.as_int())
            .ok_or_else(|| "expected int value".to_string())?;
        Ok(NativeCallResult::Return(Value::Int(
            base * input * multiplier,
        )))
    });

    Ok(())
}

#[no_mangle]
pub extern "C" fn lust_extension_register(vm_ptr: *mut VM) -> bool {
    if vm_ptr.is_null() {
        return false;
    }

    let vm = unsafe { &mut *vm_ptr };
    register_functions(vm).is_ok()
}

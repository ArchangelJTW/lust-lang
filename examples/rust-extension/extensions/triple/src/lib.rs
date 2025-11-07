use lust::{
    ast::{Span, Type, TypeKind},
    bytecode::{NativeCallResult, Value},
    embed::{enum_variant, enum_variant_with, function_param, ExternRegistry, FunctionBuilder},
    NativeExport, NativeExportParam, VM,
};

fn ty_int() -> Type {
    Type::new(TypeKind::Int, Span::dummy())
}

fn ty_operation() -> Type {
    Type::new(
        TypeKind::Named("Operation".to_string()),
        Span::dummy(),
    )
}

fn register_types(vm: &mut VM) {
    use lust::embed::native_types::EnumBuilder;

    let mut registry = ExternRegistry::new();
    let operation_enum = EnumBuilder::new("Operation")
        .variant(enum_variant("Double"))
        .variant(enum_variant("Triple"))
        .variant(enum_variant_with("Scale", [ty_int()]))
        .finish();
    registry.add_enum(operation_enum);

    let make_scale = FunctionBuilder::new("make_scale")
        .param(function_param("factor", ty_int()))
        .return_type(ty_operation())
        .finish();
    registry.add_function(make_scale);

    let apply_operation = FunctionBuilder::new("apply_operation")
        .param(function_param("operation", ty_operation()))
        .param(function_param("value", ty_int()))
        .return_type(ty_int())
        .finish();
    registry.add_function(apply_operation);

    registry.register_with_vm(vm);
}

fn register_functions(vm: &mut VM) -> Result<(), String> {
    register_types(vm);

    let export = NativeExport::new(
        "host_triple",
        vec![NativeExportParam::new("value", "int")],
        "int",
    );
    vm.register_exported_native(export, |values: &[Value]| {
        let number = values
            .get(0)
            .and_then(|value| value.as_int())
            .ok_or_else(|| "expected int argument".to_string())?;
        Ok(NativeCallResult::Return(Value::Int(number * 3)))
    });

    let export = NativeExport::new(
        "make_scale",
        vec![NativeExportParam::new("factor", "int")],
        "Operation",
    );
    vm.register_exported_native(export, |values: &[Value]| {
        let factor = values
            .get(0)
            .and_then(|value| value.as_int())
            .ok_or_else(|| "expected scale factor".to_string())?;
        Ok(NativeCallResult::Return(Value::enum_variant(
            "externs.lust_triple.Operation",
            "Scale",
            vec![Value::Int(factor)],
        )))
    });

    let export = NativeExport::new(
        "apply_operation",
        vec![
            NativeExportParam::new("operation", "Operation"),
            NativeExportParam::new("value", "int"),
        ],
        "int",
    );
    vm.register_exported_native(export, |values: &[Value]| {
        let operation = values
            .get(0)
            .ok_or_else(|| "expected operation".to_string())?;
        let input = values
            .get(1)
            .and_then(|value| value.as_int())
            .ok_or_else(|| "expected int value".to_string())?;

        let result = match operation {
            Value::Enum {
                enum_name,
                variant,
                values: payload,
            } if enum_name == "externs.lust_triple.Operation" || enum_name == "Operation" => match variant.as_str() {
                "Double" => input * 2,
                "Triple" => input * 3,
                "Scale" => {
                    let factor = payload
                        .as_ref()
                        .and_then(|values| values.get(0))
                        .and_then(|value| value.as_int())
                        .ok_or_else(|| "Scale variant requires factor".to_string())?;
                    input * factor
                }
                other => {
                    return Err(format!("Unknown Operation variant '{}'", other));
                }
            },
            other => {
                return Err(format!(
                    "Expected externs.lust_triple.Operation but received {:?}",
                    other
                ));
            }
        };

        Ok(NativeCallResult::Return(Value::Int(result)))
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

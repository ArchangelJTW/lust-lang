use lust::{
    bytecode::{NativeCallResult, Value},
    NativeExport, NativeExportParam, VM,
};

fn register_functions(vm: &mut VM) {
    let export = NativeExport::new(
        "host_double",
        vec![NativeExportParam::new("value", "int")],
        "int",
    );

    vm.register_exported_native(
        export,
        |values: &[Value]| -> Result<NativeCallResult, String> {
            let value = values
                .get(0)
                .ok_or_else(|| "expected one argument".to_string())?;
            let number = match value {
                Value::Int(v) => *v,
                other => {
                    return Err(format!("expected int but received {:?}", other));
                }
            };
            Ok(NativeCallResult::Return(Value::Int(number * 2)))
        },
    );

    let export = NativeExport::new(
        "nested.host_quadruple",
        vec![NativeExportParam::new("value", "int")],
        "int",
    );

    vm.register_exported_native(
        export,
        |values: &[Value]| -> Result<NativeCallResult, String> {
            let value = values
                .get(0)
                .ok_or_else(|| "expected one argument".to_string())?;
            let number = match value {
                Value::Int(v) => *v,
                other => {
                    return Err(format!("expected int but received {:?}", other));
                }
            };
            Ok(NativeCallResult::Return(Value::Int(number * 4)))
        },
    );
}

#[no_mangle]
pub extern "C" fn lust_extension_register(vm_ptr: *mut VM) -> bool {
    if vm_ptr.is_null() {
        return false;
    }

    let vm = unsafe { &mut *vm_ptr };
    register_functions(vm);
    true
}

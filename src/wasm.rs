#![cfg(target_arch = "wasm32")]
use crate::bytecode::{NativeCallResult, Value};
use crate::embed::EmbeddedBuilder;
use crate::LustError;
use std::cell::RefCell;
use std::fmt::Write;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub fn run_lust(source: &str) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let mut program = EmbeddedBuilder::new()
        .with_base_dir("web")
        .module("main", source)
        .entry_module("main")
        .compile()
        .map_err(to_js_error)?;
    let buffer = Rc::new(RefCell::new(String::new()));
    install_print_hooks(program.vm_mut(), buffer.clone());
    program.run_entry_script().map_err(to_js_error)?;
    let output = buffer.borrow().clone();
    Ok(output)
}

fn install_print_hooks(vm: &mut crate::vm::VM, buffer: Rc<RefCell<String>>) {
    vm.register_native("print", make_print_closure(buffer.clone(), false));
    vm.register_native("println", make_print_closure(buffer, true));
}

fn make_print_closure(buffer: Rc<RefCell<String>>, newline: bool) -> Value {
    Value::NativeFunction(Rc::new(move |args: &[Value]| {
        let mut text = String::new();
        for (index, value) in args.iter().enumerate() {
            if index > 0 {
                text.push('\t');
            }

            write!(text, "{}", value).map_err(|err| err.to_string())?;
        }

        if newline {
            text.push('\n');
        }

        {
            buffer.borrow_mut().push_str(&text);
        }

        if !text.is_empty() {
            web_sys::console::log_1(&JsValue::from_str(&text));
        }

        Ok(NativeCallResult::Return(Value::Nil))
    }))
}

fn to_js_error(err: LustError) -> JsValue {
    JsValue::from_str(&err.to_string())
}

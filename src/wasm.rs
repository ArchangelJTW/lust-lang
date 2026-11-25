//! WebAssembly and WASI runtime integration for Lust.
//!
//! Provides architectural definitions, ABI contracts, and browser-specific
//! wasm-bindgen bindings when targeting `wasm32-unknown-unknown`.

use core::fmt;

/// Target WebAssembly architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WasmArchitecture {
    Wasm32,
    Wasm64,
}

impl WasmArchitecture {
    pub fn pointer_width(&self) -> usize {
        match self {
            Self::Wasm32 => 32,
            Self::Wasm64 => 64,
        }
    }
}

/// WebAssembly system interface and execution ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WasmAbi {
    /// Browser JavaScript host environment via wasm-bindgen.
    BrowserJs,
    /// WASI Preview 1 (`wasi_snapshot_preview1`).
    WasiSnapshotPreview1,
    /// WASI Preview 2 Component Model (`wasi:cli`, `wasi:filesystem`, etc.).
    WasiPreview2,
    /// Standalone WebAssembly with no host ABI.
    BareWasm,
}

impl fmt::Display for WasmAbi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BrowserJs => write!(f, "browser-js"),
            Self::WasiSnapshotPreview1 => write!(f, "wasi-snapshot-preview1"),
            Self::WasiPreview2 => write!(f, "wasi-preview2"),
            Self::BareWasm => write!(f, "bare-wasm"),
        }
    }
}

/// Architectural metadata for the active or targeted WebAssembly platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmTarget {
    pub arch: WasmArchitecture,
    pub abi: WasmAbi,
    pub memory_page_size: usize,
}

impl Default for WasmTarget {
    fn default() -> Self {
        Self {
            arch: WasmArchitecture::Wasm32,
            abi: detect_current_abi(),
            memory_page_size: 65536, // Standard WebAssembly 64KiB page size
        }
    }
}

/// Detects the target WASM ABI based on compile-time target attributes.
pub fn detect_current_abi() -> WasmAbi {
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        WasmAbi::BrowserJs
    }
    #[cfg(target_os = "wasi")]
    {
        #[cfg(feature = "wasi-p2")]
        {
            WasmAbi::WasiPreview2
        }
        #[cfg(not(feature = "wasi-p2"))]
        {
            WasmAbi::WasiSnapshotPreview1
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        WasmAbi::BareWasm
    }
}

/// Check if the current compilation target is WebAssembly.
pub fn is_wasm_target() -> bool {
    cfg!(target_arch = "wasm32")
}

/// Runtime configuration for WebAssembly sandbox execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmRuntimeConfig {
    pub initial_memory_pages: u32,
    pub max_memory_pages: Option<u32>,
    pub fuel_limit: Option<u64>,
    pub deterministic_clock: bool,
}

impl Default for WasmRuntimeConfig {
    fn default() -> Self {
        Self {
            initial_memory_pages: 16,     // 1 MiB default initial memory
            max_memory_pages: Some(1024), // 64 MiB maximum memory
            fuel_limit: Some(1_000_000),  // 1M fuel default quantum
            deterministic_clock: true,
        }
    }
}

// =============================================================================
// Browser JavaScript Bindings (target_os = "unknown" on wasm32)
// =============================================================================

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub use browser_impl::run_lust;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod browser_impl {
    use super::*;
    use crate::LustError;
    use crate::bytecode::{NativeCallResult, Value, native_fn};
    use crate::embed::EmbeddedBuilder;
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
        Value::NativeFunction(native_fn(move |args: &[Value]| {
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
}

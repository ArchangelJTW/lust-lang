#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! DST Gate 3: Module Loader & Isolation Test Suite.
//! Verifies that module resolution, data imports, and compilation execute
//! completely inside `MemFs` without touching host disk.

use lust::config::LustConfig;
use lust::modules::{ModuleLoader, compile_program_with_config};
use lust::vfs::{MemFs, VirtualFileSystem};
use lust::Value;
use std::path::Path;
use std::sync::Arc;

#[test]
fn test_module_loading_pure_memfs() {
    let mem_fs = Arc::new(MemFs::new());

    // 1. Prepare virtual workspace modules
    mem_fs.create_dir_all(Path::new("pkg")).expect("create pkg dir");

    let math_utils_source = r#"
function add(a: int, b: int): int
    return a + b
end

function mul(a: int, b: int): int
    return a * b
end
"#;
    mem_fs
        .write(Path::new("pkg/math_utils.lust"), math_utils_source.as_bytes())
        .expect("write math_utils");

    let entry_source = r#"
use math_utils.{add, mul}

function compute(x: int, y: int): int
    return mul(add(x, y), 2)
end
"#;
    mem_fs
        .write(Path::new("pkg/main.lust"), entry_source.as_bytes())
        .expect("write main");

    // 2. Assert host disk DOES NOT have these files
    assert!(!Path::new("/workspace/pkg/main.lust").exists());
    assert!(!Path::new("pkg/main.lust").exists());

    // 3. Load program using ModuleLoader backed by MemFs
    let mut loader = ModuleLoader::new(Path::new("/workspace/pkg")).with_vfs(mem_fs.clone());
    let program = loader
        .load_program_from_entry("/workspace/pkg/main.lust")
        .expect("load program from MemFs");

    // 4. Compile program to VM
    let config = LustConfig::default();
    let mut vm = compile_program_with_config(program, &config).expect("compile program");
    vm.vfs = mem_fs.clone();

    // 5. Call public function and verify calculation parity
    let res = vm
        .call("main.compute", vec![Value::Int(5), Value::Int(3)])
        .expect("call compute");

    assert_eq!(res, Value::Int((5 + 3) * 2));
}

#[test]
fn test_module_data_import_pure_memfs() {
    let mem_fs = Arc::new(MemFs::new());

    // JSON configuration file inside MemFs
    let json_data = r#"{"factor": 10, "offset": 5}"#;
    mem_fs
        .write(Path::new("settings.json"), json_data.as_bytes())
        .expect("write json");

    let entry_source = r#"
import settings from "./settings.json"

function apply_factor(val: int): int
    return val * settings["factor"] + settings["offset"]
end
"#;
    mem_fs
        .write(Path::new("entry.lust"), entry_source.as_bytes())
        .expect("write entry");

    let mut loader = ModuleLoader::new(Path::new("/workspace")).with_vfs(mem_fs.clone());
    let program = loader
        .load_program_from_entry("/workspace/entry.lust")
        .expect("load entry with data import from MemFs");

    let config_path = Path::new("/workspace/settings.json");
    let cache = loader.data_cache();
    assert_eq!(cache.len(), 1);
    assert!(cache.contains_key(config_path));

    let config = LustConfig::default();
    let mut vm = compile_program_with_config(program, &config).expect("compile");
    vm.vfs = mem_fs.clone();

    // Run script initialization to bind top-level data imports
    let _ = vm.call("__script", vec![]);

    let res = vm
        .call("entry.apply_factor", vec![Value::Int(7)])
        .expect("call apply_factor");

    assert_eq!(res, Value::Int(7 * 10 + 5));
}

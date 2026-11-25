#![cfg(feature = "std")]

//! Master Gate-SCHEMA Integration Suite: Gate-SCHEMA-01 through Gate-SCHEMA-13.
//!
//! Conforms to RFC First-Class Data Formats and Schemas (§13 Acceptance Gates):
//! - Each test exercises the existing behavior for the corresponding gate.
//! - A gate that cannot be exercised fails the test with an explicit message (no stub pass).

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use lust::config::LustConfig;
use lust::data::bson::BsonDocumentView;
use lust::data::json::{self, JsonError};
use lust::data::jsonc;
use lust::data::jsonl::{self, JsonlReaderOptions, LineErrorKind};
use lust::data::layout::{CanonicalTypeTag, LogicalFieldDesc, LogicalLayoutV1, TargetAbi};
use lust::data::materialize::{materialize_from_document, MaterializeError};
use lust::data::owner::{LeaseError, OwnerRegistry};
use lust::data::toon::{self, ToonError};
use lust::data::yaml::{self, YamlError};
use lust::data::{BudgetTracker, DataValue, Document, ExactNumber};
use lust::modules::ModuleLoader;
use lust::regex::{compile_bounded, BoundedRegex, RegexError};

fn temp_test_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut dir = std::env::temp_dir();
    dir.push(format!("{prefix}_{nanos}"));
    dir
}

// ============================================================================
// Gate-SCHEMA-01: Semantic rejection
// ============================================================================
#[test]
fn test_gate_schema_01_semantic_rejection() {
    // 1. JSON duplicate keys strictly rejected
    assert!(
        matches!(json::from_str(r#"{"k": 1, "k": 2}"#), Err(JsonError::DuplicateKey(..))),
        "JSON duplicate keys must fail closed"
    );

    // 2. YAML duplicate keys strictly rejected
    assert!(
        matches!(yaml::from_str("k: 1\nk: 2"), Err(YamlError::DuplicateKey(..))),
        "YAML duplicate keys must fail closed"
    );

    // 3. TOON duplicate columns strictly rejected
    assert!(
        matches!(toon::from_str("[col, dup, dup]\n1, 2, 3"), Err(ToonError::DuplicateColumn(..))),
        "TOON duplicate columns must fail closed"
    );

    // 4. Document duplicate keys strictly rejected
    let mut doc = Document::new();
    doc.try_insert("id".to_string(), DataValue::Number(ExactNumber::Int(1))).unwrap();
    assert!(
        doc.try_insert("id".to_string(), DataValue::Number(ExactNumber::Int(2))).is_err(),
        "Document duplicate keys must fail closed"
    );

    // 5. Non-finite numbers rejected (JSON and TOON)
    assert!(
        matches!(json::from_str("1e999"), Err(JsonError::InvalidNumber(..))),
        "JSON overflowing float must fail closed"
    );
    assert!(
        toon::from_str("val\n1e999").is_err(),
        "TOON overflowing float must fail closed"
    );

    // 6. Integer overflow narrowing strictly rejected (fail closed, never narrowed to f64)
    assert!(
        json::from_str("18446744073709551616").is_err(),
        "JSON integer token exceeding 64-bit lane must reject and not narrow to f64"
    );
    assert!(
        toon::from_str("val\n9223372036854775808").is_err(),
        "TOON integer token exceeding i64 lane must reject and not narrow to f64"
    );

    // 7. YAML unconsumed malformed trailing junk rejected
    assert!(
        yaml::from_str("a: 1\n- malformed-suffix").is_err(),
        "YAML trailing junk must fail closed"
    );
}

// ============================================================================
// Gate-SCHEMA-02: Bounded hostile input
// ============================================================================
#[test]
fn test_gate_schema_02_bounded_hostile_input() {
    // 1. Budget tracker quota enforcement (quota - 1 ok, quota ok, quota + 1 fails closed)
    let mut tracker = BudgetTracker::new(100, 1000, 10);
    assert!(tracker.charge_bytes(99).is_ok()); // quota - 1
    assert!(tracker.charge_bytes(1).is_ok());  // quota
    assert!(tracker.charge_bytes(1).is_err(), "quota + 1 must fail closed");

    // 2. YAML billion laughs / alias bomb amplification rejected
    let mut bomb = String::from("root: &b a\n");
    for i in 0..25 {
        bomb.push_str(&format!("k{}: *b\n", i));
    }
    assert!(
        yaml::from_str_with_aliases(&bomb).is_err(),
        "YAML alias amplification exceeding budget must fail closed"
    );

    // 3. BSON nested document claiming bytes past parent boundary rejected
    let bytes = vec![
        11, 0, 0, 0, // parent declared length: 11 bytes
        0x03, b'n', 0,
        20, 0, 0, 0, // child declared length: 20 bytes (exceeds parent!)
        0x10, b'x', 0, 1, 0, 0, 0,
        0, 0,
    ];
    assert!(
        BsonDocumentView::from_slice(&bytes).is_err(),
        "BSON nested document crossing parent boundary must fail closed"
    );
}

// ============================================================================
// Gate-SCHEMA-03: Lifetime / ABI isolation
// ============================================================================
#[test]
fn test_gate_schema_03_lifetime_abi_isolation() {
    let mut reg = OwnerRegistry::new(1);
    let id = reg.allocate(1);
    reg.cancel(id).unwrap();

    // Release before join MUST fail closed with ReleaseBeforeJoin
    assert_eq!(
        reg.release(id),
        Err(LeaseError::ReleaseBeforeJoin),
        "Releasing without acknowledge_join must fail closed"
    );

    // Acknowledge join then release succeeds
    reg.acknowledge_join(id).unwrap();
    assert_eq!(reg.release(id), Ok(()));
}

// ============================================================================
// Gate-SCHEMA-04: Opt-in / DCE / authority
// ============================================================================
#[test]
fn test_gate_schema_04_opt_in_dce_authority() {
    use lust::codegen_zig::{analyze_runtime_requirements, tree_shake_items, ZigCodegen};
    use object::{Object, ObjectSymbol};

    let dir = temp_test_dir("gate04_authority");
    fs::create_dir_all(&dir).unwrap();

    let mut strict_config = LustConfig::pure();
    strict_config.set_strict_capabilities(true);

    // Baseline config checks
    assert!(
        !strict_config.is_std_module_allowed("@std/os"),
        "Unauthorized os module must be denied in strict mode"
    );
    assert!(
        !strict_config.is_std_module_allowed("@std/io"),
        "Unauthorized io module must be denied in strict mode"
    );

    // 1. Compile errors for unpermitted std module (@std/os or @std/io):
    // 1a. Import: import statement & use statement
    let import_file = dir.join("deny_import.lust");
    fs::write(
        &import_file,
        "import os from \"@std/os\"\nfunction main(): int return 0 end\n",
    )
    .unwrap();
    let mut loader = ModuleLoader::new(dir.clone()).with_config(strict_config.clone());
    let res_import = loader.load_program_from_entry(import_file.to_str().unwrap());
    assert!(
        res_import.is_err(),
        "import of unpermitted @std/os must be a compile error"
    );

    let use_io_file = dir.join("deny_use_io.lust");
    fs::write(
        &use_io_file,
        "use std.io\nfunction main(): int return 0 end\n",
    )
    .unwrap();
    let mut loader = ModuleLoader::new(dir.clone()).with_config(strict_config.clone());
    let res_use_io = loader.load_program_from_entry(use_io_file.to_str().unwrap());
    assert!(
        res_use_io.is_err(),
        "use import of unpermitted std.io must be a compile error"
    );

    // 1b. Alias: use ... as ...
    let alias_os_file = dir.join("deny_alias_os.lust");
    fs::write(
        &alias_os_file,
        "use std.os as my_os\nfunction main(): int return 0 end\n",
    )
    .unwrap();
    let mut loader = ModuleLoader::new(dir.clone()).with_config(strict_config.clone());
    let res_alias = loader.load_program_from_entry(alias_os_file.to_str().unwrap());
    assert!(
        res_alias.is_err(),
        "alias of unpermitted std.os must be a compile error"
    );

    let alias_io_file = dir.join("deny_alias_io.lust");
    fs::write(
        &alias_io_file,
        "use std.io as my_io\nfunction main(): int return 0 end\n",
    )
    .unwrap();
    let mut loader = ModuleLoader::new(dir.clone()).with_config(strict_config.clone());
    let res_alias_io = loader.load_program_from_entry(alias_io_file.to_str().unwrap());
    assert!(
        res_alias_io.is_err(),
        "alias of unpermitted std.io must be a compile error"
    );

    // 1c. Re-export: export import ... from "@std/..."
    let reexport_os_file = dir.join("deny_reexport_os.lust");
    fs::write(
        &reexport_os_file,
        "export import os from \"@std/os\"\nfunction main(): int return 0 end\n",
    )
    .unwrap();
    let mut loader = ModuleLoader::new(dir.clone()).with_config(strict_config.clone());
    let res_reexport_os = loader.load_program_from_entry(reexport_os_file.to_str().unwrap());
    assert!(
        res_reexport_os.is_err(),
        "re-export of unpermitted std.os must be a compile error"
    );

    let reexport_io_file = dir.join("deny_reexport_io.lust");
    fs::write(
        &reexport_io_file,
        "export import io from \"@std/io\"\nfunction main(): int return 0 end\n",
    )
    .unwrap();
    let mut loader = ModuleLoader::new(dir.clone()).with_config(strict_config.clone());
    let res_reexport_io = loader.load_program_from_entry(reexport_io_file.to_str().unwrap());
    assert!(
        res_reexport_io.is_err(),
        "re-export of unpermitted std.io must be a compile error"
    );

    // Same re-export form must compile when module is allowed
    let mut permitted_reexport_config = LustConfig::pure();
    permitted_reexport_config.set_strict_capabilities(true);
    permitted_reexport_config.allow_std_module("os");
    let mut loader_allowed_reexport =
        ModuleLoader::new(dir.clone()).with_config(permitted_reexport_config);
    let prog_allowed_reexport = loader_allowed_reexport
        .load_program_from_entry(reexport_os_file.to_str().unwrap())
        .expect("re-export of permitted std.os must compile successfully");
    assert!(
        prog_allowed_reexport.modules[0]
            .exports
            .functions
            .contains_key("os"),
        "re-exported module alias must be registered in exports"
    );

    // 1d. Dynamic import: import("@std/os") or import("@std/io")
    let dynamic_os_file = dir.join("deny_dyn_os.lust");
    fs::write(
        &dynamic_os_file,
        "function main(): int\n    local m = import(\"@std/os\")\n    return 0\nend\n",
    )
    .unwrap();
    let mut loader = ModuleLoader::new(dir.clone()).with_config(strict_config.clone());
    let res_dyn_os = loader.load_program_from_entry(dynamic_os_file.to_str().unwrap());
    assert!(
        res_dyn_os.is_err(),
        "dynamic import of unpermitted @std/os must be a compile error"
    );

    let dynamic_io_file = dir.join("deny_dyn_io.lust");
    fs::write(
        &dynamic_io_file,
        "function main(): int\n    local m = import(\"@std/io\")\n    return 0\nend\n",
    )
    .unwrap();
    let mut loader = ModuleLoader::new(dir.clone()).with_config(strict_config.clone());
    let res_dyn_io = loader.load_program_from_entry(dynamic_io_file.to_str().unwrap());
    assert!(
        res_dyn_io.is_err(),
        "dynamic import of unpermitted @std/io must be a compile error"
    );

    // 2. The denial happens before any host effect of that module runs. Assert effect was not observed.
    let witness_file = dir.join("host_effect_observed.txt");
    let effect_file = dir.join("deny_effect.lust");
    fs::write(
        &effect_file,
        format!(
            "import os from \"@std/os\"\nfunction main(): int\n    os.create_file(\"{}\")\n    return 0\nend\n",
            witness_file.display()
        ),
    )
    .unwrap();
    let mut loader = ModuleLoader::new(dir.clone()).with_config(strict_config.clone());
    let res_effect = loader.load_program_from_entry(effect_file.to_str().unwrap());
    assert!(
        res_effect.is_err(),
        "compilation must fail before host effect can run"
    );
    assert!(
        !witness_file.exists(),
        "host effect must not be observed when unpermitted: witness file was created!"
    );

    // 2b. Positive execution check: When @std/os is allowed, execute the program and show witness file is created.
    let mut permitted_os_config = LustConfig::pure();
    permitted_os_config.set_strict_capabilities(true);
    permitted_os_config.allow_std_module("os");
    let mut loader_permitted_os =
        ModuleLoader::new(dir.clone()).with_config(permitted_os_config.clone());
    let prog_effect = loader_permitted_os
        .load_program_from_entry(effect_file.to_str().unwrap())
        .expect("permitted program must load successfully");
    let entry_fn = format!("{}.main", prog_effect.entry_module);
    let mut vm_effect = lust::compile_program_with_config(prog_effect, &permitted_os_config)
        .expect("permitted program must compile to VM");
    if vm_effect.call("main", vec![]).is_err() {
        let _ = vm_effect.call(&entry_fn, vec![]).expect("running main must succeed");
    }
    assert!(
        witness_file.exists(),
        "witness file must be created when @std/os is permitted and executed"
    );
    let _ = fs::remove_file(&witness_file);

    // 3. Permitted module (@std/bson after an explicit allow) still imports
    let mut permitted_config = LustConfig::pure();
    permitted_config.set_strict_capabilities(true);
    permitted_config.allow_std_module("bson");
    assert!(
        permitted_config.is_std_module_allowed("@std/bson"),
        "Permitted module must pass authorization check"
    );

    let bson_file = dir.join("allow_bson.lust");
    fs::write(
        &bson_file,
        "import bson from \"@std/bson\"\nfunction main(): int return 42 end\n",
    )
    .unwrap();
    let mut loader_permitted = ModuleLoader::new(dir.clone()).with_config(permitted_config);
    let prog_bson = loader_permitted
        .load_program_from_entry(bson_file.to_str().unwrap())
        .expect("permitted @std/bson must import successfully");
    assert_eq!(prog_bson.modules.len(), 1);
    assert!(
        prog_bson.modules[0]
            .imports
            .module_aliases
            .contains_key("bson"),
        "@std/bson alias must be registered on permitted import"
    );

    // 4. Dead-code elimination: compiled artifact and link output do not contain denied host entries
    let dce_file = dir.join("dce_program.lust");
    fs::write(
        &dce_file,
        "function compute(x: int): int\n    return x * 2 + 1\nend\nlocal r: int = compute(21)\n",
    )
    .unwrap();
    let mut loader_dce = ModuleLoader::new(dir.clone()).with_config(strict_config.clone());
    let prog_dce = loader_dce
        .load_program_from_entry(dce_file.to_str().unwrap())
        .expect("pure program must compile successfully");

    let shaken = tree_shake_items(&prog_dce.modules[0].items, false);
    let opts = analyze_runtime_requirements(&shaken, false, None);
    assert!(
        !opts.needs_print && !opts.needs_os && !opts.needs_io,
        "print/os/io runtime must not be needed for pure compute"
    );

    let mut codegen = ZigCodegen::with_options(opts);
    let compiled_artifact = codegen
        .generate_program(&shaken)
        .expect("AOT codegen must succeed");

    // Check compiled source artifact directly
    assert!(
        !compiled_artifact.contains("__lust_print"),
        "compiled artifact must not contain __lust_print"
    );
    assert!(
        !compiled_artifact.contains("__lust_println"),
        "compiled artifact must not contain __lust_println"
    );
    assert!(
        !compiled_artifact.contains("std.c.write"),
        "compiled artifact must not contain host std.c.write entry"
    );
    assert!(
        !compiled_artifact.contains("fwrite"),
        "compiled artifact must not contain host fwrite entry"
    );
    assert!(
        !compiled_artifact.contains("os.remove_file"),
        "compiled artifact must not contain host os.remove_file entry"
    );
    assert!(
        !compiled_artifact.contains("unlink"),
        "compiled artifact must not contain host unlink entry"
    );

    // Compile pure binary / link output artifact
    let zig_path = dir.join("dce_artifact.zig");
    let obj_path = dir.join("dce_artifact.o");
    fs::write(&zig_path, &compiled_artifact).unwrap();
    let obj_output = std::process::Command::new("zig")
        .arg("build-obj")
        .arg(&zig_path)
        .arg(format!("-femit-bin={}", obj_path.display()))
        .output()
        .expect("invoke zig build-obj to produce pure link artifact");
    assert!(
        obj_output.status.success(),
        "zig build-obj must succeed for pure artifact: {}",
        String::from_utf8_lossy(&obj_output.stderr)
    );

    // 4b. Positive control: A program that calls the host entry must contain the actual emitted symbol
    let host_file = dir.join("host_program.lust");
    fs::write(
        &host_file,
        "use std.os\nfunction effect(): bool\n    return os.remove_file(\"tmp.txt\")\nend\nlocal r: bool = effect()\n",
    )
    .unwrap();
    let mut loader_host =
        ModuleLoader::new(dir.clone()).with_config(permitted_os_config.clone());
    let prog_host = loader_host
        .load_program_from_entry(host_file.to_str().unwrap())
        .expect("host program must load");
    let shaken_host = tree_shake_items(&prog_host.modules[0].items, false);
    let opts_host = analyze_runtime_requirements(&shaken_host, false, None);
    assert!(
        opts_host.needs_os,
        "host program must require os runtime"
    );
    let mut codegen_host = ZigCodegen::with_options(opts_host);
    let compiled_host = codegen_host
        .generate_program(&shaken_host)
        .expect("AOT codegen must succeed for host program");

    let host_zig_path = dir.join("host_artifact.zig");
    let host_obj_path = dir.join("host_artifact.o");
    fs::write(&host_zig_path, &compiled_host).unwrap();
    let host_obj_output = std::process::Command::new("zig")
        .arg("build-obj")
        .arg("-lc")
        .arg(&host_zig_path)
        .arg(format!("-femit-bin={}", host_obj_path.display()))
        .output()
        .expect("invoke zig build-obj to produce host link artifact");
    assert!(
        host_obj_output.status.success(),
        "zig build-obj must succeed for host artifact: {}",
        String::from_utf8_lossy(&host_obj_output.stderr)
    );

    let host_obj_bytes = fs::read(&host_obj_path).expect("read host binary artifact");
    let host_obj_file = object::File::parse(&*host_obj_bytes).expect("parse host binary artifact");

    // The positive control must contain the emitted symbol (mangled remove_file or C linkage unlink)
    let host_symbol_name = host_obj_file
        .symbols()
        .map(|s| s.name().unwrap_or_default().to_string())
        .find(|name| name.contains("remove_file") || name.strip_prefix('_').unwrap_or(name) == "unlink")
        .expect("positive control must contain emitted host entry symbol (remove_file or unlink)");

    // Check pure compiled binary artifact: must lack that exact symbol and host entry symbols
    let obj_bytes = fs::read(&obj_path).expect("read compiled binary artifact");
    let obj_file = object::File::parse(&*obj_bytes).expect("parse compiled binary artifact");
    let pure_symbols: std::collections::HashSet<String> = obj_file
        .symbols()
        .map(|s| s.name().unwrap_or_default().to_string())
        .collect();

    assert!(
        !pure_symbols.contains(&host_symbol_name),
        "compiled link artifact must lack positive control host entry symbol '{}'",
        host_symbol_name
    );

    for sym in obj_file.symbols() {
        let name = sym.name().unwrap_or_default();
        assert!(
            !name.contains("remove_file")
                && !name.contains("fwrite")
                && !name.contains("write_stdout")
                && !name.contains("write_file")
                && name.strip_prefix('_').unwrap_or(name) != "unlink",
            "compiled link artifact must not contain host entry: {}",
            name
        );
    }

    let _ = fs::remove_dir_all(dir);
}

// ============================================================================
// Gate-SCHEMA-05: Backend/effect/cost parity
// ============================================================================
#[test]
fn test_gate_schema_05_backend_effect_cost_parity() {
    use lust::ast::ItemKind;
    use lust::bytecode::Value;
    use lust::effect_abi::LUST_STATUS_ERR_OUT_OF_GAS;
    use lust::intern::Interner;
    use lust::lexer::Lexer;
    use lust::mir::cranelift_backend::jit_engine::{
        CraneliftJitCompiler, lust_mir_get_gas, lust_mir_get_memory_used, lust_mir_get_status,
        lust_mir_set_gas, lust_mir_set_status,
    };
    use lust::mir::ir::MirModule;
    use lust::mir::lowering::AstLowering;
    use lust::parser::Parser;
    use lust::vm::VM;

    // The Lust program runs a real effect handler:
    // Declares an effect signature `! string`, constructs a real effect block via `effect do ... end`,
    // and executes it via `.run_sync()`.
    let program_src = r#"
        function compute_with_effect(n: int, fail_flag: int): int ! string
            local eff = effect do
                if fail_flag != 0 then
                    return -1
                end
                local arr = [10, 20, 30]
                local sum: int = 0
                local i: int = 0
                while i < n do
                    sum = sum + arr[0]:unwrap()
                    i = i + 1
                end
                return sum
            end
            return eff:run_sync()
        end
    "#;

    let mut interner = Interner::new();
    let mut lexer = Lexer::new(program_src, &mut interner);
    let mut parser = Parser::from_lexer(&mut lexer).expect("lexer must tokenize program");
    let items = parser.parse().expect("parser must parse program items");

    let mut func_def_opt = None;
    for item in &items {
        if let ItemKind::Function(func_def) = &item.kind {
            func_def_opt = Some(func_def.clone());
            break;
        }
    }
    let func_def = func_def_opt.expect("Function compute_with_effect must be parsed");

    // ------------------------------------------------------------------------
    // Backend 1: Bytecode VM
    // Runs the Lust program with the real effect handler.
    // ------------------------------------------------------------------------
    let mut compiler = lust::Compiler::new();
    let bytecode_fns = compiler.compile_module(&items)
        .expect("Bytecode VM backend must compile program with effect handler");
    let mut vm = VM::new();
    vm.load_functions(bytecode_fns);

    let vm_success_val = vm.call("compute_with_effect", vec![Value::Int(5), Value::Int(0)])
        .expect("Bytecode VM execution of compute_with_effect must succeed");
    let vm_success = match vm_success_val {
        Value::Int(v) => v,
        other => panic!("Bytecode VM returned unexpected value: {:?}", other),
    };
    assert_eq!(
        vm_success, 50,
        "Bytecode VM must produce expected result 50 on success path"
    );

    let vm_fail_val = vm.call("compute_with_effect", vec![Value::Int(5), Value::Int(1)])
        .expect("Bytecode VM execution of compute_with_effect failure branch must return value");
    let vm_fail = match vm_fail_val {
        Value::Int(v) => v,
        other => panic!("Bytecode VM returned unexpected value: {:?}", other),
    };
    assert_eq!(
        vm_fail, -1,
        "Bytecode VM must produce expected failure result -1"
    );

    // ------------------------------------------------------------------------
    // Backend 2: Cranelift Backend
    // A backend that cannot run must fail the test, not be skipped.
    // Lower function into MIR (preserving `! string`).
    // If Cranelift cannot run the handler, the test must fail.
    // ------------------------------------------------------------------------
    let mir_fn = match AstLowering::lower_function(&func_def) {
        Ok(func) => {
            assert!(
                func.signature.effects.is_some(),
                "MIR lowering must not drop effect signature (! string)"
            );
            func
        }
        Err(err) => {
            panic!(
                "Gate-SCHEMA-05 FAIL: Cranelift backend cannot run effect handler (MIR lowering failed: {err}). \
                 Per RFC §13 Acceptance Gates, a gate that cannot be exercised must fail the test with an explicit message."
            );
        }
    };

    let mut module = MirModule::new("gate05_parity_mod".into());
    module.add_function(mir_fn.clone());

    let mut jit = CraneliftJitCompiler::new()
        .expect("Cranelift JIT backend must initialize");
    let code_ptr = match jit.compile_function(&mir_fn) {
        Ok(ptr) => ptr,
        Err(err) => {
            panic!(
                "Gate-SCHEMA-05 FAIL: Cranelift backend cannot run effect handler (Cranelift JIT compilation failed: {err}). \
                 Per RFC §13 Acceptance Gates, a gate that cannot be exercised must fail the test with an explicit message."
            );
        }
    };
    let clif_func: extern "C" fn(i64, i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };

    // Cranelift parity checks (if Cranelift could compile and run the handler):
    lust_mir_set_gas(1_000_000);
    lust_mir_set_status(0);
    let clif_success = clif_func(5, 0);
    assert_eq!(
        lust_mir_get_status(), 0,
        "Cranelift status must be 0 on normal success call"
    );
    assert!(
        lust_mir_get_gas() >= 0,
        "Cranelift remaining gas must be nonnegative on normal success call"
    );
    assert_eq!(
        clif_success, 50,
        "Cranelift backend must produce expected result 50 on success path"
    );
    assert_eq!(
        vm_success, clif_success,
        "Bytecode VM and Cranelift backend must produce identical results on success path"
    );

    lust_mir_set_gas(1_000_000);
    lust_mir_set_status(0);
    let clif_fail = clif_func(5, 1);
    assert_eq!(
        lust_mir_get_status(), 0,
        "Cranelift status must be 0 on normal failure branch call"
    );
    assert!(
        lust_mir_get_gas() >= 0,
        "Cranelift remaining gas must be nonnegative on normal failure branch call"
    );
    assert_eq!(
        clif_fail, -1,
        "Cranelift backend must produce expected failure result -1"
    );
    assert_eq!(
        vm_fail, clif_fail,
        "Bytecode VM and Cranelift backend must produce identical results on failure branch"
    );

    // Ample-budget execution control of (1000, 0) on both backends
    vm.clear_gas_budget();
    let vm_ample_val = vm.call("compute_with_effect", vec![Value::Int(1000), Value::Int(0)])
        .expect("Bytecode VM execution with ample budget must succeed");
    let vm_ample = match vm_ample_val {
        Value::Int(v) => v,
        other => panic!("Bytecode VM returned unexpected value: {:?}", other),
    };
    assert_eq!(
        vm_ample, 10000,
        "Bytecode VM must produce expected result 10000 on ample budget"
    );

    lust_mir_set_gas(10_000_000);
    lust_mir_set_status(0);
    let clif_ample = clif_func(1000, 0);
    assert_eq!(
        lust_mir_get_status(), 0,
        "Cranelift status must be 0 on ample-budget call"
    );
    assert!(
        lust_mir_get_gas() >= 0,
        "Cranelift remaining gas must be nonnegative on ample-budget call"
    );
    assert_eq!(
        clif_ample, 10000,
        "Cranelift backend must produce expected result 10000 on ample budget"
    );
    assert_eq!(
        vm_ample, clif_ample,
        "Bytecode VM and Cranelift backend must produce identical results on ample budget"
    );

    // ------------------------------------------------------------------------
    // 2. Interrupting that effect on each backend yields the same error and
    //    does not return a success value.
    //    Delete hardcoded Exit literals. Assert on the error each backend actually returns.
    //    Cranelift must not look successful by returning 0 while a side status flag is interpreted.
    // ------------------------------------------------------------------------
    let expected_success_val: i64 = 10_000;

    // Interrupt Bytecode VM via constrained gas budget
    vm.set_gas_budget(2);
    let vm_interrupt_res = vm.call(
        "compute_with_effect",
        vec![Value::Int(1000), Value::Int(0)],
    );
    assert!(
        vm_interrupt_res.is_err(),
        "Bytecode VM must yield an error when interrupted, but returned: {:?}",
        vm_interrupt_res
    );
    let vm_err_str = vm_interrupt_res.as_ref().unwrap_err().to_string();
    assert!(
        vm_err_str.to_lowercase().contains("out of gas") || vm_err_str.contains("interrupted"),
        "Bytecode VM error must indicate gas exhaustion or interruption: {}",
        vm_err_str
    );

    // Interrupt Cranelift backend via constrained gas budget
    lust_mir_set_gas(2);
    lust_mir_set_status(0);
    let clif_interrupt_val = clif_func(1000, 0);
    let clif_gas_rem = lust_mir_get_gas();
    let clif_status = lust_mir_get_status();

    assert_ne!(
        clif_interrupt_val, expected_success_val,
        "Cranelift backend must not return a success value when interrupted (returned {})",
        clif_interrupt_val
    );
    assert_ne!(
        clif_interrupt_val, 0,
        "Cranelift backend must not return 0 (looking successful) on gas exhaustion",
    );
    assert_eq!(
        clif_interrupt_val, -(LUST_STATUS_ERR_OUT_OF_GAS as i64),
        "Cranelift backend must return negative out-of-gas status code on gas exhaustion",
    );
    assert_eq!(
        clif_status, LUST_STATUS_ERR_OUT_OF_GAS as i64,
        "Cranelift status flag must record LUST_STATUS_ERR_OUT_OF_GAS (status={})",
        clif_status
    );
    assert!(
        clif_gas_rem < 0,
        "Cranelift backend must record gas exhaustion (gas_rem={})",
        clif_gas_rem
    );

    // ------------------------------------------------------------------------
    // 3. A counted allocation or gas budget is the same order on both backends
    //    for that program: the larger figure is at most 2x the smaller one.
    //    Print both figures in the assertion message. Keep dynamic allocation counts.
    // ------------------------------------------------------------------------
    vm.clear_gas_budget();
    vm.set_memory_budget_bytes(1024 * 1024).expect("memory budget config must succeed");
    vm.reset_memory_counter().expect("memory counter reset must succeed");
    let vm_alloc_val = vm.call("compute_with_effect", vec![Value::Int(10), Value::Int(0)])
        .expect("VM execution for allocation measurement must succeed");
    let vm_alloc_res = match vm_alloc_val {
        Value::Int(v) => v,
        other => panic!("Bytecode VM returned unexpected value: {:?}", other),
    };
    assert_eq!(
        vm_alloc_res, 100,
        "VM execution for allocation measurement must return 100"
    );
    let vm_allocated_bytes = vm.memory_used_bytes();

    lust_mir_set_gas(1_000_000);
    lust_mir_set_status(0);
    let clif_mem_before = lust_mir_get_memory_used();
    let clif_alloc_res = clif_func(10, 0);
    assert_eq!(
        lust_mir_get_status(), 0,
        "Cranelift status must be 0 on allocation measurement call"
    );
    assert!(
        lust_mir_get_gas() >= 0,
        "Cranelift remaining gas must be nonnegative on allocation measurement call"
    );
    assert_eq!(
        clif_alloc_res, 100,
        "Cranelift execution for allocation measurement must return 100"
    );
    let clif_allocated_bytes = (lust_mir_get_memory_used() - clif_mem_before) as usize;

    assert!(
        vm_allocated_bytes > 0,
        "Bytecode VM must account for array allocation (recorded 0 bytes)"
    );
    assert!(
        clif_allocated_bytes > 0,
        "Cranelift backend must account for array allocation (recorded 0 bytes)"
    );

    let max_alloc = vm_allocated_bytes.max(clif_allocated_bytes);
    let min_alloc = vm_allocated_bytes.min(clif_allocated_bytes);
    eprintln!(
        "Gate-05 allocation: vm={vm_allocated_bytes} cranelift={clif_allocated_bytes}"
    );

    assert!(
        max_alloc <= 2 * min_alloc,
        "Counted allocation cost parity assertion failed: larger ({}) must be at most 2x smaller ({}) [Bytecode VM allocated {} bytes, Cranelift allocated {} bytes]",
        max_alloc,
        min_alloc,
        vm_allocated_bytes,
        clif_allocated_bytes
    );
}

fn parse_gate05_fn(src: &str) -> (lust::ast::FunctionDef, Vec<lust::ast::Item>) {
    let mut interner = lust::intern::Interner::new();
    let mut lexer = lust::lexer::Lexer::new(src, &mut interner);
    let mut parser = lust::parser::Parser::from_lexer(&mut lexer).expect("lexer must tokenize program");
    let items = parser.parse().expect("parser must parse program items");
    let mut func_def_opt = None;
    for item in &items {
        if let lust::ast::ItemKind::Function(func_def) = &item.kind {
            func_def_opt = Some(func_def.clone());
            break;
        }
    }
    (func_def_opt.expect("Function must be parsed"), items)
}

fn compile_gate05_vm(items: &[lust::ast::Item]) -> lust::vm::VM {
    let mut compiler = lust::Compiler::new();
    let bytecode_fns = compiler.compile_module(items)
        .expect("Bytecode VM backend must compile program");
    let mut vm = lust::vm::VM::new();
    vm.load_functions(bytecode_fns);
    vm
}

// ----------------------------------------------------------------------------
// Gate-05 focused tests
// ----------------------------------------------------------------------------

#[test]
fn test_gate_schema_05_uninvoked_thunk_is_lazy() {
    use lust::bytecode::Value;
    use lust::mir::cranelift_backend::jit_engine::{
        CraneliftJitCompiler, lust_mir_get_gas, lust_mir_get_memory_used, lust_mir_get_status,
        lust_mir_set_gas, lust_mir_set_status,
    };
    use lust::mir::lowering::AstLowering;

    let ctrl_src = r#"
        function control(): int
            return 123
        end
    "#;

    let uninvoked_src = r#"
        function uninvoked(): int
            local eff = effect do
                local arr = [7, 8, 9]
                return arr[0]:unwrap()
            end
            return 123
        end
    "#;

    let invoked_src = r#"
        function invoked(): int
            local eff = effect do
                local arr = [7, 8, 9]
                return arr[0]:unwrap()
            end
            return eff:run_sync()
        end
    "#;

    // 1. Control execution (no thunk, returns 123)
    let (ctrl_def, ctrl_items) = parse_gate05_fn(ctrl_src);
    let mut ctrl_vm = compile_gate05_vm(&ctrl_items);
    ctrl_vm.clear_gas_budget();
    ctrl_vm.set_memory_budget_bytes(1024 * 1024).unwrap();
    ctrl_vm.reset_memory_counter().unwrap();
    let vm_ctrl_val = ctrl_vm.call("control", vec![]).unwrap();
    assert_eq!(vm_ctrl_val, Value::Int(123));
    let vm_ctrl_alloc = ctrl_vm.memory_used_bytes();

    let mir_ctrl = AstLowering::lower_function(&ctrl_def).expect("MIR lowering must succeed");
    let mut jit_ctrl = CraneliftJitCompiler::new().expect("JIT init must succeed");
    let code_ptr_ctrl = jit_ctrl.compile_function(&mir_ctrl).expect("JIT compile must succeed");
    let clif_ctrl_fn: extern "C" fn() -> i64 = unsafe { core::mem::transmute(code_ptr_ctrl) };
    lust_mir_set_gas(1_000_000);
    lust_mir_set_status(0);
    let clif_mem_before = lust_mir_get_memory_used();
    let clif_ctrl_val = clif_ctrl_fn();
    assert_eq!(lust_mir_get_status(), 0);
    assert!(lust_mir_get_gas() >= 0);
    assert_eq!(clif_ctrl_val, 123);
    let clif_ctrl_alloc = (lust_mir_get_memory_used() - clif_mem_before) as usize;

    // 2. Uninvoked execution (thunk constructed with array body, returns 123 without invoking)
    let (uninvoked_def, uninvoked_items) = parse_gate05_fn(uninvoked_src);
    let mut uninvoked_vm = compile_gate05_vm(&uninvoked_items);
    uninvoked_vm.clear_gas_budget();
    uninvoked_vm.set_memory_budget_bytes(1024 * 1024).unwrap();
    uninvoked_vm.reset_memory_counter().unwrap();
    let vm_uninvoked_val = uninvoked_vm.call("uninvoked", vec![]).unwrap();
    assert_eq!(vm_uninvoked_val, Value::Int(123));
    let vm_uninvoked_alloc = uninvoked_vm.memory_used_bytes();

    let mir_uninvoked = AstLowering::lower_function(&uninvoked_def).expect("MIR lowering must succeed");
    let mut jit_uninvoked = CraneliftJitCompiler::new().expect("JIT init must succeed");
    let code_ptr_uninvoked = jit_uninvoked.compile_function(&mir_uninvoked).expect("JIT compile must succeed");
    let clif_uninvoked_fn: extern "C" fn() -> i64 = unsafe { core::mem::transmute(code_ptr_uninvoked) };
    lust_mir_set_gas(1_000_000);
    lust_mir_set_status(0);
    let clif_mem_before = lust_mir_get_memory_used();
    let clif_uninvoked_val = clif_uninvoked_fn();
    assert_eq!(lust_mir_get_status(), 0);
    assert!(lust_mir_get_gas() >= 0);
    assert_eq!(clif_uninvoked_val, 123);
    let clif_uninvoked_alloc = (lust_mir_get_memory_used() - clif_mem_before) as usize;

    // Parity check on return values:
    assert_eq!(vm_uninvoked_val, Value::Int(clif_uninvoked_val));

    // Measured allocation attributable to the body is zero:
    let vm_body_delta = vm_uninvoked_alloc.saturating_sub(vm_ctrl_alloc);
    assert_eq!(
        vm_body_delta, 0,
        "VM body allocation for uninvoked thunk must be 0 (uninvoked={vm_uninvoked_alloc}, ctrl={vm_ctrl_alloc})"
    );
    let clif_body_delta = clif_uninvoked_alloc.saturating_sub(clif_ctrl_alloc);
    assert_eq!(
        clif_body_delta, 0,
        "Cranelift body allocation for uninvoked thunk must be 0 (uninvoked={clif_uninvoked_alloc}, ctrl={clif_ctrl_alloc})"
    );

    // 3. Invoked execution (positive control: returns 7 with positive allocation increase over uninvoked)
    let (invoked_def, invoked_items) = parse_gate05_fn(invoked_src);
    let mut invoked_vm = compile_gate05_vm(&invoked_items);
    invoked_vm.clear_gas_budget();
    invoked_vm.set_memory_budget_bytes(1024 * 1024).unwrap();
    invoked_vm.reset_memory_counter().unwrap();
    let vm_invoked_val = invoked_vm.call("invoked", vec![]).unwrap();
    assert_eq!(vm_invoked_val, Value::Int(7));
    let vm_invoked_alloc = invoked_vm.memory_used_bytes();

    let mir_invoked = AstLowering::lower_function(&invoked_def).expect("MIR lowering must succeed");
    let mut jit_invoked = CraneliftJitCompiler::new().expect("JIT init must succeed");
    let code_ptr_invoked = jit_invoked.compile_function(&mir_invoked).expect("JIT compile must succeed");
    let clif_invoked_fn: extern "C" fn() -> i64 = unsafe { core::mem::transmute(code_ptr_invoked) };
    lust_mir_set_gas(1_000_000);
    lust_mir_set_status(0);
    let clif_mem_before = lust_mir_get_memory_used();
    let clif_invoked_val = clif_invoked_fn();
    assert_eq!(lust_mir_get_status(), 0);
    assert!(lust_mir_get_gas() >= 0);
    assert_eq!(clif_invoked_val, 7);
    let clif_invoked_alloc = (lust_mir_get_memory_used() - clif_mem_before) as usize;

    assert_eq!(vm_invoked_val, Value::Int(clif_invoked_val));
    assert!(
        vm_invoked_alloc > vm_uninvoked_alloc,
        "VM invoked allocation ({vm_invoked_alloc}) must be greater than uninvoked ({vm_uninvoked_alloc})"
    );
    assert!(
        clif_invoked_alloc > clif_uninvoked_alloc,
        "Cranelift invoked allocation ({clif_invoked_alloc}) must be greater than uninvoked ({clif_uninvoked_alloc})"
    );
}

#[test]
fn test_gate_schema_05_capture_snapshot() {
    use lust::bytecode::Value;
    use lust::mir::cranelift_backend::jit_engine::{
        CraneliftJitCompiler, lust_mir_get_gas, lust_mir_get_status, lust_mir_set_gas,
        lust_mir_set_status,
    };
    use lust::mir::lowering::AstLowering;

    let src_reassign = r#"
        function capture_reassign(x: int): int
            local eff = effect do
                return x
            end
            x = 99
            return eff:run_sync() + x
        end
    "#;

    let src_shadow = r#"
        function capture_shadow(x: int): int
            local eff = effect do
                return x
            end
            local x: int = 99
            return eff:run_sync() + x
        end
    "#;

    for (name, src) in [("capture_reassign", src_reassign), ("capture_shadow", src_shadow)] {
        let (func_def, items) = parse_gate05_fn(src);

        // Bytecode VM
        let mut vm = compile_gate05_vm(&items);
        vm.clear_gas_budget();
        let vm_res_val = vm.call(name, vec![Value::Int(7)])
            .expect("VM execution must succeed");
        let vm_res = match vm_res_val {
            Value::Int(v) => v,
            other => panic!("VM returned unexpected: {:?}", other),
        };

        // Cranelift
        let mir_fn = AstLowering::lower_function(&func_def)
            .expect("MIR lowering must succeed");
        let mut jit = CraneliftJitCompiler::new()
            .expect("Cranelift JIT backend must initialize");
        let code_ptr = jit.compile_function(&mir_fn)
            .expect("Cranelift JIT compilation must succeed");
        let clif_fn: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };

        lust_mir_set_gas(1_000_000);
        lust_mir_set_status(0);
        let clif_res = clif_fn(7);
        assert_eq!(lust_mir_get_status(), 0);
        assert!(lust_mir_get_gas() >= 0);

        assert_eq!(clif_res, 106, "{name} on Cranelift must be 106, got {clif_res}");
        assert_eq!(vm_res, 106, "{name} on VM must be 106, got {vm_res}");
    }
}

#[test]
fn test_gate_schema_05_thunk_return_resumes_caller() {
    use lust::bytecode::Value;
    use lust::mir::cranelift_backend::jit_engine::{
        CraneliftJitCompiler, lust_mir_get_gas, lust_mir_get_status, lust_mir_set_gas,
        lust_mir_set_status,
    };
    use lust::mir::lowering::AstLowering;

    let src = r#"
        function thunk_return_resumes(): int
            local eff = effect do
                return 7
            end
            local res: int = eff:run_sync()
            return res + 5
        end
    "#;

    let (func_def, items) = parse_gate05_fn(src);

    // Bytecode VM
    let mut vm = compile_gate05_vm(&items);
    vm.clear_gas_budget();
    let vm_res_val = vm.call("thunk_return_resumes", vec![])
        .expect("VM execution must succeed");
    let vm_res = match vm_res_val {
        Value::Int(v) => v,
        other => panic!("VM returned unexpected: {:?}", other),
    };
    assert_eq!(vm_res, 12, "Bytecode VM must resume caller after thunk return, expecting 12");

    // Cranelift
    let mir_fn = AstLowering::lower_function(&func_def)
        .expect("MIR lowering must succeed");
    let mut jit = CraneliftJitCompiler::new()
        .expect("Cranelift JIT backend must initialize");
    let code_ptr = jit.compile_function(&mir_fn)
        .expect("Cranelift JIT compilation must succeed");
    let clif_fn: extern "C" fn() -> i64 = unsafe { core::mem::transmute(code_ptr) };

    lust_mir_set_gas(1_000_000);
    lust_mir_set_status(0);
    let clif_res = clif_fn();
    assert_eq!(lust_mir_get_status(), 0);
    assert!(lust_mir_get_gas() >= 0);
    assert_eq!(clif_res, 12, "Cranelift must resume caller after thunk return, expecting 12");
    assert_eq!(vm_res, clif_res, "Both backends must agree on resumed result 12");
}

#[test]
fn test_gate_schema_05_repeated_invocation_reruns_body() {
    use lust::bytecode::Value;
    use lust::mir::cranelift_backend::jit_engine::{
        CraneliftJitCompiler, lust_mir_get_gas, lust_mir_get_memory_used, lust_mir_get_status,
        lust_mir_set_gas, lust_mir_set_status,
    };
    use lust::mir::lowering::AstLowering;

    let src = r#"
        function repeated_invocations(count: int): int
            local eff = effect do
                local arr = [7, 8, 9]
                return arr[0]:unwrap()
            end
            local sum: int = 0
            local i: int = 0
            while i < count do
                sum = sum + eff:run_sync()
                i = i + 1
            end
            return sum
        end
    "#;

    let (func_def, items) = parse_gate05_fn(src);

    let mir_fn = AstLowering::lower_function(&func_def)
        .expect("MIR lowering must succeed");
    let mut jit = CraneliftJitCompiler::new()
        .expect("Cranelift JIT backend must initialize");
    let code_ptr = jit.compile_function(&mir_fn)
        .expect("Cranelift JIT compilation must succeed");
    let clif_fn: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(code_ptr) };

    let mut vm_costs = Vec::new();
    let mut clif_costs = Vec::new();

    let expected_results = [0, 7, 14];

    for count in 0..=2 {
        // VM run
        let mut vm = compile_gate05_vm(&items);
        vm.clear_gas_budget();
        vm.set_memory_budget_bytes(1024 * 1024).unwrap();
        vm.reset_memory_counter().unwrap();
        let vm_res_val = vm.call("repeated_invocations", vec![Value::Int(count)])
            .expect("VM execution must succeed");
        let vm_res = match vm_res_val {
            Value::Int(v) => v,
            other => panic!("VM returned unexpected: {:?}", other),
        };
        assert_eq!(
            vm_res, expected_results[count as usize],
            "VM result for count {count} must be {}", expected_results[count as usize]
        );
        vm_costs.push(vm.memory_used_bytes());

        // Cranelift run
        lust_mir_set_gas(1_000_000);
        lust_mir_set_status(0);
        let mem_before = lust_mir_get_memory_used();
        let clif_res = clif_fn(count);
        assert_eq!(lust_mir_get_status(), 0);
        assert!(lust_mir_get_gas() >= 0);
        assert_eq!(
            clif_res, expected_results[count as usize],
            "Cranelift result for count {count} must be {}", expected_results[count as usize]
        );
        clif_costs.push((lust_mir_get_memory_used() - mem_before) as usize);

        assert_eq!(vm_res, clif_res, "Parity between VM and Cranelift for count {count}");
    }

    // Allocation assertions per backend:
    // D1 > D0 and D2 - D1 == D1 - D0
    let (v0, v1, v2) = (vm_costs[0], vm_costs[1], vm_costs[2]);
    assert!(v1 > v0, "VM D1 ({v1}) must be greater than D0 ({v0})");
    assert_eq!(v2 - v1, v1 - v0, "VM D2 - D1 ({}) must equal D1 - D0 ({})", v2 - v1, v1 - v0);

    let (c0, c1, c2) = (clif_costs[0], clif_costs[1], clif_costs[2]);
    assert!(c1 > c0, "Cranelift D1 ({c1}) must be greater than D0 ({c0})");
    assert_eq!(c2 - c1, c1 - c0, "Cranelift D2 - D1 ({}) must equal D1 - D0 ({})", c2 - c1, c1 - c0);
}

#[test]
fn test_gate_schema_05_unsupported_escape_fails_closed() {
    use lust::mir::lowering::{AstLowering, MirLowerError};

    let src = r#"
        function escape_thunk(): int
            local eff = effect do
                return 42
            end
            return eff
        end
    "#;

    let (func_def, _items) = parse_gate05_fn(src);

    let lower_res = AstLowering::lower_function(&func_def);
    match lower_res {
        Err(MirLowerError::UnlowerableExpression(msg)) => {
            assert!(
                msg.contains("cannot escape or be used as a value"),
                "Diagnostic must mention thunk escape/value rejection, got: {msg}"
            );
        }
        other => {
            panic!("Expected Err(MirLowerError::UnlowerableExpression) for escaping thunk, got: {other:?}");
        }
    }
}

// ============================================================================
// Gate-SCHEMA-06: Streaming JSONL Memory Cap
// ============================================================================

#[derive(Clone, Default)]
struct Gate06ReadTracker {
    bytes_read: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    read_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Gate06ReadTracker {
    fn new() -> Self {
        Self::default()
    }

    fn bytes_read(&self) -> usize {
        self.bytes_read.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn read_calls(&self) -> usize {
        self.read_calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Finite lazy std::io::Read fixture emitting repeated bytes without allocating the complete stream.
struct LazyHostileLineReader {
    pos: usize,
    total_len: usize,
    lf_pos: Option<usize>,
    max_emit_per_read: usize,
    tracker: Gate06ReadTracker,
}

impl LazyHostileLineReader {
    fn new(total_len: usize, lf_pos: Option<usize>, tracker: Gate06ReadTracker) -> Self {
        Self {
            pos: 0,
            total_len,
            lf_pos,
            max_emit_per_read: jsonl::DEFAULT_CHUNK_SIZE,
            tracker,
        }
    }
}

impl std::io::Read for LazyHostileLineReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.total_len || buf.is_empty() {
            return Ok(0);
        }
        let remaining = self.total_len - self.pos;
        let to_emit = buf.len().min(remaining).min(self.max_emit_per_read);
        if to_emit == 0 {
            return Ok(0);
        }

        buf[..to_emit].fill(b'a');
        if let Some(nl) = self.lf_pos {
            if nl >= self.pos && nl < self.pos + to_emit {
                buf[nl - self.pos] = b'\n';
            }
        }

        self.pos += to_emit;
        self.tracker.bytes_read.fetch_add(to_emit, std::sync::atomic::Ordering::SeqCst);
        self.tracker.read_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(to_emit)
    }
}

/// Finite lazy std::io::Read fixture emitting small valid JSONL records on the fly without preallocation.
struct LazyValidRecordsReader {
    current_index: usize,
    total_records: usize,
    buffer: [u8; 64],
    buf_pos: usize,
    buf_len: usize,
    tracker: Gate06ReadTracker,
}

impl LazyValidRecordsReader {
    fn new(total_records: usize, tracker: Gate06ReadTracker) -> Self {
        Self {
            current_index: 0,
            total_records,
            buffer: [0u8; 64],
            buf_pos: 0,
            buf_len: 0,
            tracker,
        }
    }
}

impl std::io::Read for LazyValidRecordsReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut written = 0;
        while written < buf.len() {
            if self.buf_pos < self.buf_len {
                let avail = self.buf_len - self.buf_pos;
                let needed = buf.len() - written;
                let to_copy = avail.min(needed);
                buf[written..written + to_copy].copy_from_slice(&self.buffer[self.buf_pos..self.buf_pos + to_copy]);
                self.buf_pos += to_copy;
                written += to_copy;
            } else if self.current_index < self.total_records {
                use std::io::Write;
                let mut cursor = std::io::Cursor::new(&mut self.buffer[..]);
                write!(cursor, "{{\"id\":{}}}\n", self.current_index).unwrap();
                self.buf_len = cursor.position() as usize;
                self.buf_pos = 0;
                self.current_index += 1;
            } else {
                break;
            }
        }
        if written > 0 {
            self.tracker.bytes_read.fetch_add(written, std::sync::atomic::Ordering::SeqCst);
            self.tracker.read_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        Ok(written)
    }
}

fn assert_hostile_line_case(
    total_len: usize,
    lf_pos: Option<usize>,
    case_desc: &str,
) -> (usize, usize) {
    let line_limit = 16 * 1024;
    let buffered_limit = 32 * 1024;
    let tracker = Gate06ReadTracker::new();
    let fixture = LazyHostileLineReader::new(total_len, lf_pos, tracker.clone());

    let opts = JsonlReaderOptions::default()
        .with_max_line_bytes(line_limit)
        .with_max_buffered_bytes(buffered_limit);

    let mut reader = jsonl::JsonlReader::with_options(fixture, opts);

    // 1. Assert effective options match requested
    assert_eq!(
        reader.options().max_line_bytes,
        line_limit,
        "{case_desc}: effective max_line_bytes"
    );
    assert_eq!(
        reader.options().max_buffered_bytes,
        buffered_limit,
        "{case_desc}: effective max_buffered_bytes"
    );

    // 2. Observe capacity at construction
    let cap_init = reader.retained_buffer_bytes();
    assert!(
        cap_init <= 32 * 1024,
        "{case_desc}: initial retained buffer {cap_init} > 32 KiB"
    );
    let mut max_cap = cap_init;

    // 3. Pull first item: must fail closed with LineErrorKind::Limit and LINE_TOO_LONG
    let first = reader.next();
    assert!(first.is_some(), "{case_desc}: stream should yield an error entry");
    let err = first.unwrap().expect_err(&format!("{case_desc}: expected error entry"));
    assert_eq!(
        err.kind,
        LineErrorKind::Limit,
        "{case_desc}: error kind must be Limit"
    );
    assert_eq!(
        err.code,
        "LINE_TOO_LONG",
        "{case_desc}: error code must be LINE_TOO_LONG"
    );

    // 4. Retained capacity immediately after error must remain bounded
    let cap_err = reader.retained_buffer_bytes();
    max_cap = max_cap.max(cap_err);
    assert!(
        cap_err <= 32 * 1024,
        "{case_desc}: retained buffer after error {cap_err} > 32 KiB"
    );

    // 5. Delivered bytes bound: delivered bytes <= line_limit + DEFAULT_CHUNK_SIZE
    let delivered_bytes = tracker.bytes_read();
    let max_allowed_bytes = line_limit + jsonl::DEFAULT_CHUNK_SIZE;
    assert!(
        delivered_bytes <= max_allowed_bytes,
        "{case_desc}: delivered bytes {delivered_bytes} exceeded line_limit + DEFAULT_CHUNK_SIZE ({max_allowed_bytes})"
    );

    // 6. Repeated next() returns None and does NOT perform additional reads
    let reads_after_err = tracker.read_calls();
    let bytes_after_err = tracker.bytes_read();

    let term1 = reader.next();
    assert!(term1.is_none(), "{case_desc}: subsequent next() must be None");
    assert_eq!(
        tracker.read_calls(),
        reads_after_err,
        "{case_desc}: terminal next() must not perform I/O reads"
    );
    assert_eq!(
        tracker.bytes_read(),
        bytes_after_err,
        "{case_desc}: terminal next() must not read bytes"
    );

    let cap_term1 = reader.retained_buffer_bytes();
    max_cap = max_cap.max(cap_term1);
    assert!(
        cap_term1 <= 32 * 1024,
        "{case_desc}: retained buffer after term1 {cap_term1} > 32 KiB"
    );

    let term2 = reader.next();
    assert!(term2.is_none(), "{case_desc}: second terminal next() must be None");
    assert_eq!(
        tracker.read_calls(),
        reads_after_err,
        "{case_desc}: second terminal next() must not perform I/O"
    );
    assert_eq!(
        tracker.bytes_read(),
        bytes_after_err,
        "{case_desc}: second terminal next() must not read bytes"
    );

    let cap_term2 = reader.retained_buffer_bytes();
    max_cap = max_cap.max(cap_term2);
    assert!(
        cap_term2 <= 32 * 1024,
        "{case_desc}: retained buffer after term2 {cap_term2} > 32 KiB"
    );

    println!(
        "[{case_desc}] observed max capacity: {} bytes, delivered: {} bytes, reads: {}",
        max_cap,
        delivered_bytes,
        reads_after_err
    );

    (max_cap, delivered_bytes)
}

#[test]
fn test_gate_schema_06_streaming_jsonl_memory_cap() {
    // 1. Hostile lines at 1 MiB and 16 MiB, no LF and far-end LF, plus LF in crossing chunk
    let (cap_1m_nolf, bytes_1m_nolf) = assert_hostile_line_case(
        1024 * 1024,
        None,
        "Hostile 1 MiB no-LF",
    );
    let (cap_1m_farend, bytes_1m_farend) = assert_hostile_line_case(
        1024 * 1024,
        Some(1024 * 1024 - 1),
        "Hostile 1 MiB far-end LF",
    );
    let (cap_16m_nolf, bytes_16m_nolf) = assert_hostile_line_case(
        16 * 1024 * 1024,
        None,
        "Hostile 16 MiB no-LF",
    );
    let (cap_16m_farend, bytes_16m_farend) = assert_hostile_line_case(
        16 * 1024 * 1024,
        Some(16 * 1024 * 1024 - 1),
        "Hostile 16 MiB far-end LF",
    );
    let (cap_cross_lf, bytes_cross_lf) = assert_hostile_line_case(
        1024 * 1024,
        Some(16 * 1024 + 100),
        "Hostile 1 MiB LF in crossing chunk",
    );

    // Enforce that observed capacity bounds are identical (<= 32 KiB) across both advertised lengths
    let hostile_max_cap = cap_1m_nolf
        .max(cap_1m_farend)
        .max(cap_16m_nolf)
        .max(cap_16m_farend)
        .max(cap_cross_lf);
    assert!(
        hostile_max_cap <= 32 * 1024,
        "Max observed hostile capacity {hostile_max_cap} exceeds 32 KiB bound"
    );
    assert_eq!(
        cap_1m_nolf, cap_16m_nolf,
        "Observed capacity must be invariant to total advertised length"
    );
    assert_eq!(
        bytes_1m_nolf, bytes_16m_nolf,
        "Delivered bytes must be invariant to total advertised length"
    );
    assert_eq!(
        bytes_1m_nolf, bytes_1m_farend,
        "Delivered bytes must be invariant to LF position at far end (1 MiB)"
    );
    assert_eq!(
        bytes_16m_nolf, bytes_16m_farend,
        "Delivered bytes must be invariant to LF position at far end (16 MiB)"
    );
    assert_eq!(
        bytes_1m_nolf, bytes_cross_lf,
        "Delivered bytes must be invariant when LF is located in crossing chunk"
    );

    // 2. Original short-line Limit case
    {
        let mut opts = JsonlReaderOptions::default();
        opts.max_line_bytes = 32;

        let long_line = format!(r#"{{"long_field": "{}"}}"#, "a".repeat(100));
        let mut reader = jsonl::JsonlReader::with_options(long_line.as_bytes(), opts);
        let first = reader.next();
        assert!(first.is_some(), "stream should yield an error entry");
        let res = first.unwrap();
        assert!(
            matches!(res, Err(ref e) if matches!(e.kind, LineErrorKind::Limit)),
            "oversized line must fail closed with Limit error"
        );
        let err = res.as_ref().unwrap_err();
        assert_eq!(err.code, "LINE_TOO_LONG");
        assert!(reader.next().is_none(), "next after limit must be None");
    }

    // 3. Lazy stream of small valid records, one pull at a time
    {
        let total_records = 2000;
        let tracker = Gate06ReadTracker::new();
        let fixture = LazyValidRecordsReader::new(total_records, tracker.clone());
        let opts = JsonlReaderOptions::default()
            .with_max_line_bytes(16 * 1024)
            .with_max_buffered_bytes(32 * 1024);
        let mut reader = jsonl::JsonlReader::with_options(fixture, opts);

        assert_eq!(reader.options().max_line_bytes, 16 * 1024);
        assert_eq!(reader.options().max_buffered_bytes, 32 * 1024);

        let mut max_valid_cap = reader.retained_buffer_bytes();
        assert!(max_valid_cap <= 32 * 1024);

        let mut record_count = 0;
        while let Some(res) = reader.next() {
            let val = res.expect("valid JSONL record expected");
            if let DataValue::Document(doc) = &val {
                let id_val = doc.get("id").and_then(|v| v.as_i64());
                assert_eq!(
                    id_val,
                    Some(record_count as i64),
                    "record id must match sequential index"
                );
            } else {
                panic!("expected Document record");
            }
            record_count += 1;

            // Retained buffer capacity sampled after every pull must stay within 32 KiB
            let current_cap = reader.retained_buffer_bytes();
            max_valid_cap = max_valid_cap.max(current_cap);
            assert!(
                current_cap <= 32 * 1024,
                "Retained buffer {} exceeded 32 KiB at record {}",
                current_cap,
                record_count
            );

            // Verify inspecting state between pulls causes no reads
            let reads_before = tracker.read_calls();
            let _ = reader.retained_buffer_bytes();
            let _ = reader.live_memory_bytes();
            assert_eq!(
                tracker.read_calls(),
                reads_before,
                "inspecting state between pulls must cause no additional reads"
            );

            // Dropped each record explicitly at end of pull iteration
            drop(val);
        }

        assert_eq!(record_count, total_records);
        assert!(reader.next().is_none(), "stream at EOF yields None");

        // Sample at EOF
        let eof_cap = reader.retained_buffer_bytes();
        max_valid_cap = max_valid_cap.max(eof_cap);
        assert!(
            eof_cap <= 32 * 1024,
            "Retained buffer at EOF {} exceeded 32 KiB",
            eof_cap
        );

        println!(
            "[Gate-06 valid stream] records: {}, max retained capacity: {} bytes, total bytes read: {}",
            record_count,
            max_valid_cap,
            tracker.bytes_read()
        );
    }

    // 4. Blank-line fail-closed (valid, blank error, stop before next valid record)
    {
        let blank_stream = "{\"a\": 1}\n\n{\"b\": 2}\n";
        let mut default_reader = jsonl::JsonlReader::new(blank_stream.as_bytes());

        let first = default_reader.next();
        assert!(first.is_some(), "first record must be present");
        assert!(first.unwrap().is_ok(), "first record valid");

        let second = default_reader.next();
        assert!(second.is_some(), "blank line must emit an error");
        let err = second.unwrap().expect_err("blank line must fail closed");
        assert_eq!(
            err.kind,
            LineErrorKind::BlankLine,
            "blank line error kind must be BlankLine"
        );

        let third = default_reader.next();
        assert!(
            third.is_none(),
            "stream must terminate before the next valid record"
        );

        let fourth = default_reader.next();
        assert!(fourth.is_none(), "stream remains terminated");
    }
}

// ============================================================================
// Gate-SCHEMA-07: JSONC Comment Invariance
// ============================================================================
#[test]
fn test_gate_schema_07_jsonc_comment_invariance() {
    let jsonc_src = r#"
    // Header comment
    {
        /* block comment */
        "id": 100, // inline line comment
        "message": "/* comment inside string preserved */",
        "nested": {
            "flag": true,
        },
    }
    "#;

    let parsed = jsonc::from_str(jsonc_src).expect("JSONC parse succeeds");
    if let DataValue::Document(doc) = &parsed {
        assert_eq!(doc.get("id"), Some(&DataValue::Number(ExactNumber::Int(100))));
        assert_eq!(
            doc.get("message"),
            Some(&DataValue::String("/* comment inside string preserved */".to_string()))
        );
        if let Some(DataValue::Document(inner)) = doc.get("nested") {
            assert_eq!(inner.get("flag"), Some(&DataValue::Bool(true)));
        } else {
            panic!("nested document missing");
        }
    } else {
        panic!("expected document");
    }

    // Strict JSON must reject comments and trailing commas
    assert!(json::from_str(jsonc_src).is_err(), "strict JSON must reject JSONC syntax");
}

// ============================================================================
// Gate-SCHEMA-08: ReDoS & Regex Resource Bounding
// ============================================================================
#[test]
fn test_gate_schema_08_redos_regex_resource_bounding() {
    // 1. ReDoS backtracking and lookaround patterns rejected at compile time
    assert!(
        matches!(compile_bounded(r"(?=lookahead)test"), Err(RegexError::UnsupportedPattern(..))),
        "lookahead patterns must be rejected"
    );
    assert!(
        matches!(compile_bounded(r"([a-z])\1"), Err(RegexError::UnsupportedPattern(..))),
        "backreferences must be rejected"
    );

    // 2. Functional acceptance and rejection smoke checks (not deterministic work evidence)
    let re = BoundedRegex::new(r"^[a-z]+@[a-z]+\.[a-z]+$").expect("valid email regex");
    assert!(re.is_match("user@example.com").unwrap());
    assert!(!re.is_match("invalid-email").unwrap());

    // 3. Reported search-call counts and match results on selected adversarial inputs
    // These assertions check reported search-call counts and match results,
    // not work inside a DFA call.
    let no_match_re = BoundedRegex::new("a*z").expect("valid regex");
    let long_a = "a".repeat(20_000);
    let (found, stats) = no_match_re.find_with_stats(&long_a).expect("execution succeeds");
    assert_eq!(found, None);
    assert_eq!(stats.dfa_searches, 1);

    let match_re = BoundedRegex::new("a+b").expect("valid regex");
    for &n in &[10_000usize, 20_000usize] {
        let input = format!("{}c{}b", "a".repeat(n), "a".repeat(n));
        let (m, stats) = match_re.find_with_stats(&input).expect("execution succeeds");
        let m = m.expect("match must be present");
        assert_eq!(m.start, n + 1);
        assert_eq!(m.end, 2 * n + 2);
        assert_eq!(m.text, format!("{}b", "a".repeat(n)));
        assert_eq!(stats.dfa_searches, 2);
    }
}

// ============================================================================
// Gate-SCHEMA-09: Ordinal materialization and fail-closed validation (partial Gate-SCHEMA-09 coverage)
// ============================================================================
#[test]
fn test_gate_schema_09_typed_struct_parity_zero_cost_access() {
    // Scope: ordinal slot values, MissingField, and TypeMismatch through the Rust materialization API;
    // backend parity and zero-cost access are not established; Gate-09 remains CONCERNS;
    // the test name is not an acceptance claim.
    let fields = vec![
        LogicalFieldDesc::new(0, "user_id", CanonicalTypeTag::Int64),
        LogicalFieldDesc::new(1, "verified", CanonicalTypeTag::Bool),
    ];
    let layout = LogicalLayoutV1::new_with_abi(4001, "Account", fields, TargetAbi::x86_64());

    // 1. Valid document materializes to the expected ordinal slot values. Construction itself performs name lookups.
    let mut valid_doc = Document::new();
    valid_doc.try_insert("user_id".to_string(), DataValue::Number(ExactNumber::Int(42))).unwrap();
    valid_doc.try_insert("verified".to_string(), DataValue::Bool(true)).unwrap();

    let mat = materialize_from_document(&layout, &valid_doc).expect("materialization succeeds");
    assert_eq!(mat.get_slot(0), Some(&DataValue::Number(ExactNumber::Int(42))));
    assert_eq!(mat.get_slot(1), Some(&DataValue::Bool(true)));

    // 2. Missing required field returns MissingField instead of a materialized result.
    let mut incomplete_doc = Document::new();
    incomplete_doc.try_insert("user_id".to_string(), DataValue::Number(ExactNumber::Int(42))).unwrap();
    assert!(
        matches!(materialize_from_document(&layout, &incomplete_doc), Err(MaterializeError::MissingField { .. })),
        "missing required field must fail closed"
    );

    // 3. Type mismatch fails closed
    let mut bad_type_doc = Document::new();
    bad_type_doc.try_insert("user_id".to_string(), DataValue::String("not_int".to_string())).unwrap();
    bad_type_doc.try_insert("verified".to_string(), DataValue::Bool(true)).unwrap();
    assert!(
        matches!(materialize_from_document(&layout, &bad_type_doc), Err(MaterializeError::TypeMismatch { .. })),
        "type mismatch must fail closed"
    );
}

// ============================================================================
// Gate-SCHEMA-10: Data import reproducibility
// ============================================================================
#[test]
fn test_gate_schema_10_data_import_reproducibility() {
    let dir = temp_test_dir("gate10_data_import");
    fs::create_dir_all(&dir).unwrap();
    let config_path = dir.join("config.json");
    let entry_path = dir.join("main.lust");

    fs::write(&config_path, r#"{"database": "lust_db", "port": 5432, "ssl": true}"#).unwrap();
    fs::write(
        &entry_path,
        r#"
import config from "./config.json"

function db_port(): int
    return config["port"]
end
"#,
    )
    .unwrap();

    let mut loader = ModuleLoader::new(dir.clone());
    let prog = loader
        .load_program_from_entry(entry_path.to_str().unwrap())
        .expect("compile-time data import succeeds");
    assert_eq!(prog.modules.len(), 1);

    // Record initial cache digest and verify initial port 5432
    let (initial_digest, initial_val) = {
        let cache = loader.data_cache();
        assert_eq!(cache.len(), 1);
        let (d, v) = cache
            .get(&config_path)
            .expect("config_path present in data cache");
        let initial_bytes = fs::read(&config_path).unwrap();
        assert_eq!(*d, ModuleLoader::digest_bytes(&initial_bytes));
        let doc = v.as_document().expect("data import parsed as document");
        assert_eq!(
            doc.get("port"),
            Some(&DataValue::Number(ExactNumber::Int(5432)))
        );
        (*d, v.clone())
    };

    // Overwrite config.json with identical length, port 6432
    let new_json = r#"{"database": "lust_db", "port": 6432, "ssl": true}"#;
    fs::write(&config_path, new_json).unwrap();
    let read_back = fs::read(&config_path).unwrap();
    assert_eq!(read_back, new_json.as_bytes());
    let new_file_digest = ModuleLoader::digest_bytes(&read_back);
    assert_ne!(
        new_file_digest, initial_digest,
        "new file content must produce different digest"
    );

    // Reload via the same loader without clearing cache or creating a new loader
    let reloaded = loader
        .load_data_import(&config_path)
        .expect("data import reload succeeds");
    assert_ne!(
        reloaded, initial_val,
        "returned data value must differ from initial value"
    );
    let reloaded_doc = reloaded
        .as_document()
        .expect("reloaded data import returns document");
    assert_eq!(
        reloaded_doc.get("port"),
        Some(&DataValue::Number(ExactNumber::Int(6432))),
        "reloaded document port must be 6432"
    );

    // Cache must have one entry with the new digest and new value (port 6432)
    {
        let cache = loader.data_cache();
        assert_eq!(cache.len(), 1, "cache must still have exactly one entry");
        let (cached_digest, cached_val) = cache
            .get(&config_path)
            .expect("config_path present in data cache after reload");
        assert_eq!(
            *cached_digest, new_file_digest,
            "cached digest must match new file digest"
        );
        assert_ne!(
            *cached_digest, initial_digest,
            "cached digest must not equal old digest"
        );
        assert_eq!(
            cached_val, &reloaded,
            "cached value must equal returned new value"
        );
        assert_ne!(
            cached_val, &initial_val,
            "cached value must not equal old value"
        );
        let cached_doc = cached_val
            .as_document()
            .expect("cached value must be document");
        assert_eq!(
            cached_doc.get("port"),
            Some(&DataValue::Number(ExactNumber::Int(6432))),
            "cached document port must be 6432"
        );
    }

    // Missing file emits compile error
    let missing_path = dir.join("missing.lust");
    fs::write(&missing_path, "import bad from \"./absent.json\"\n").unwrap();
    let mut loader_missing = ModuleLoader::new(dir.clone());
    assert!(
        loader_missing.load_program_from_entry(missing_path.to_str().unwrap()).is_err(),
        "missing data file must produce compile error"
    );

    // Invalid JSON emits compile error
    let bad_json_path = dir.join("bad.json");
    fs::write(&bad_json_path, "{ broken_json: true }").unwrap();
    let bad_entry = dir.join("bad_entry.lust");
    fs::write(&bad_entry, "import x from \"./bad.json\"\n").unwrap();
    let mut loader_bad = ModuleLoader::new(dir.clone());
    assert!(
        loader_bad.load_program_from_entry(bad_entry.to_str().unwrap()).is_err(),
        "invalid JSON must produce compile error"
    );

    // Non-finite number (1e999) emits compile error
    let overflow_json_path = dir.join("overflow.json");
    fs::write(&overflow_json_path, r#"{"val": 1e999}"#).unwrap();
    let overflow_entry = dir.join("overflow_entry.lust");
    fs::write(&overflow_entry, "import x from \"./overflow.json\"\n").unwrap();
    let mut loader_overflow = ModuleLoader::new(dir.clone());
    assert!(
        loader_overflow.load_program_from_entry(overflow_entry.to_str().unwrap()).is_err(),
        "non-finite number must produce compile error"
    );

    let _ = fs::remove_dir_all(dir);
}

// ============================================================================
// Gate-SCHEMA-11: Callable semantics
// ============================================================================
#[test]
fn test_gate_schema_11_callable_semantics() {
    use lust::ast::{Expr, ExprKind, Span};
    use lust::mir::cranelift_backend::jit_engine::CraneliftJitCompiler;
    use lust::mir::cranelift_backend::MirCraneliftBackendError;
    use lust::mir::ir::{
        FunctionSignature, MirConstant, MirFunction, MirType, Operand, Terminator,
    };
    use lust::mir::lowering::{AstLowering, MirLowerError};

    // 1. AstLowering lower_expr must emit error on unlowerable bare EffectBlock in unsupported
    //    bare value context rather than silent Unit (only local bindings are supported for effect blocks).
    let dummy_expr = Expr::new(
        ExprKind::EffectBlock(vec![]), // Unsupported bare value context; only local bindings are supported
        Span::dummy(),
    );
    let mut mir_fn = MirFunction::new(
        FunctionSignature {
            name: "gate_11_bare_effect_fn".into(),
            param_types: vec![],
            return_type: MirType::Int,
            is_public: true,
            effects: None,
        },
        Span::dummy(),
    );
    let mut lowering = AstLowering::new(&mut mir_fn);
    let lower_res = lowering.lower_expr(&dummy_expr);
    assert!(
        matches!(lower_res, Err(MirLowerError::UnlowerableExpression(..))),
        "bare EffectBlock in unsupported value context must emit UnlowerableExpression error, not silent Unit"
    );
    if let Err(MirLowerError::UnlowerableExpression(msg)) = &lower_res {
        assert!(
            msg.contains("Bare EffectBlock in unsupported value context"),
            "expected diagnostic text for unsupported bare value context, got: {msg}"
        );
    }

    // 2. Cranelift MIR compiler must reject an explicit Unit operand on an Int return path with UnlowerableValue
    let sig_unit = FunctionSignature {
        name: "gate_11_reject_unit_fn".into(),
        param_types: vec![],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut mir_fn_unit = MirFunction::new(sig_unit, Span::dummy());
    let unit_entry = mir_fn_unit.entry;
    mir_fn_unit.blocks[unit_entry.0].terminator = Some(Terminator::Return(
        Some(Operand::Constant(MirConstant::Unit)),
        Span::dummy(),
    ));

    let mut jit_compiler = CraneliftJitCompiler::new().expect("fresh CraneliftJitCompiler must initialize");
    let compile_unit_res = jit_compiler.compile_function(&mir_fn_unit);
    match &compile_unit_res {
        Err(err @ MirCraneliftBackendError::UnlowerableValue(..)) => {
            assert!(
                err.to_string().contains("silent Unit"),
                "error message must explicitly declare rejection of silent Unit fallback: {err}"
            );
        }
        other => {
            panic!("expected Err(MirCraneliftBackendError::UnlowerableValue(..)), got: {other:?}");
        }
    }

    // 3. Positive control: same shape with Constant(Int(7)), fresh compiler, compilation succeeds
    let sig_pos = FunctionSignature {
        name: "gate_11_positive_int_fn".into(),
        param_types: vec![],
        return_type: MirType::Int,
        is_public: true,
        effects: None,
    };
    let mut mir_fn_int = MirFunction::new(sig_pos, Span::dummy());
    let pos_entry = mir_fn_int.entry;
    mir_fn_int.blocks[pos_entry.0].terminator = Some(Terminator::Return(
        Some(Operand::Constant(MirConstant::Int(7))),
        Span::dummy(),
    ));

    let mut jit_compiler_pos = CraneliftJitCompiler::new().expect("fresh CraneliftJitCompiler must initialize");
    let compile_pos_res = jit_compiler_pos.compile_function(&mir_fn_int);
    assert!(
        compile_pos_res.is_ok(),
        "positive control with Constant(Int(7)) must compile successfully, got: {compile_pos_res:?}"
    );
}

// ============================================================================
// Gate-SCHEMA-12: Rx lifecycle/backpressure
// ============================================================================
#[test]
fn test_gate_schema_12_rx_lifecycle_backpressure() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use lust::rx::{
        interval, BoundedQueue, BufferOptions, ExitStatus, Observable, Observer, OverflowStrategy,
        RxError, RxScheduler,
    };

    // ------------------------------------------------------------------------
    // Part 1: Full Operator Pipeline (map, filter, take)
    // Pipeline: [1..=6] -> map(*2) -> filter(divisible by 4) -> take(2) yields [4, 8]
    // ------------------------------------------------------------------------
    {
        let scheduler = Arc::new(RxScheduler::new_virtual());
        let scope_id = 1201;

        let received = Arc::new(Mutex::new(Vec::new()));
        let completed = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));

        let r_clone = received.clone();
        let c_clone = completed.clone();
        let f_clone = failed.clone();

        let observer = Observer::new(
            move |item: i64| {
                r_clone.lock().unwrap().push(item);
                Ok(())
            },
            move || {
                c_clone.store(true, Ordering::SeqCst);
            },
            move |_: RxError<String>| {
                f_clone.store(true, Ordering::SeqCst);
            },
        );

        let pipeline = Observable::<i64, String>::from_iter(vec![1, 2, 3, 4, 5, 6])
            .map(|x| x * 2)
            .filter(|x| x % 4 == 0)
            .take(2);

        let sub = pipeline
            .subscribe(
                observer,
                scope_id,
                BufferOptions::default_bounded(),
                scheduler.clone(),
            )
            .expect("Subscribe must succeed");

        scheduler.run_until_quiescent(100);

        let completion = sub
            .join(Some(Duration::from_millis(100)))
            .expect("Join must succeed");
        assert_eq!(completion.exit, ExitStatus::Completed);
        assert_eq!(completion.items_delivered, 2);
        assert_eq!(*received.lock().unwrap(), vec![4, 8]);
        assert!(completed.load(Ordering::SeqCst));
        assert!(!failed.load(Ordering::SeqCst));
    }

    // ------------------------------------------------------------------------
    // Part 2: Dual-Capacity Bounded Queue & Rejection
    // ------------------------------------------------------------------------
    {
        use lust::rx::EnqueueResult;

        #[derive(Debug, Clone, PartialEq, Eq)]
        struct Payload(usize);
        impl lust::rx::MeasurableItem for Payload {
            fn charged_bytes(&self) -> usize {
                self.0
            }
        }

        let opts = BufferOptions::new(2, 8, 1, 4, OverflowStrategy::Fail).unwrap();
        let mut q: BoundedQueue<Payload> = BoundedQueue::new(opts);

        assert_eq!(q.try_enqueue(Payload(4)), EnqueueResult::Enqueued);
        assert_eq!(q.try_enqueue(Payload(4)), EnqueueResult::Enqueued);
        // Byte and item capacity reached (2 items, 8 bytes). Next item fails backpressure.
        assert_eq!(
            q.try_enqueue(Payload(1)),
            EnqueueResult::OverflowError
        );

        // Single oversized item rejection
        let opts_large = BufferOptions::new(10, 8, 1, 4, OverflowStrategy::Fail).unwrap();
        let mut q_large: BoundedQueue<Payload> = BoundedQueue::new(opts_large);
        assert_eq!(
            q_large.try_enqueue(Payload(9)),
            EnqueueResult::ItemTooLarge
        );
    }

    // ------------------------------------------------------------------------
    // Part 3: Scope Cancellation & Zero Lingering Callbacks
    // ------------------------------------------------------------------------
    {
        let scheduler = Arc::new(RxScheduler::new_virtual());
        let scope_id = 1203;

        let received = Arc::new(Mutex::new(Vec::new()));
        let completed = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));

        let r_clone = received.clone();
        let c_clone = completed.clone();
        let f_clone = failed.clone();

        let observer = Observer::new(
            move |item: i64| {
                r_clone.lock().unwrap().push(item);
                Ok(())
            },
            move || {
                c_clone.store(true, Ordering::SeqCst);
            },
            move |_: RxError<String>| {
                f_clone.store(true, Ordering::SeqCst);
            },
        );

        let stream: Observable<i64, String> = Observable::from_producer(move |control| {
            for i in 0..100 {
                if !control.is_active() {
                    break;
                }
                if let Err(_) = control.deliver_next(i) {
                    break;
                }
            }
            control.complete();
            Ok(())
        });

        let sub = stream
            .subscribe(
                observer,
                scope_id,
                BufferOptions::default_bounded(),
                scheduler.clone(),
            )
            .unwrap();

        // Cancel subscription before running scheduler
        sub.cancel();

        scheduler.run_until_quiescent(100);

        let completion = sub.join(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(completion.exit, ExitStatus::Cancelled);
        assert!(!completed.load(Ordering::SeqCst));
        assert!(!failed.load(Ordering::SeqCst));
        assert_eq!(received.lock().unwrap().len(), 0);
    }

    // ------------------------------------------------------------------------
    // Part 4: Self-Unsubscribe & Deadlock Prevention (SelfJoin)
    // ------------------------------------------------------------------------
    {
        let scheduler = Arc::new(RxScheduler::new_virtual());
        let scope_id = 1204;

        let received = Arc::new(Mutex::new(Vec::new()));
        let self_join_detected = Arc::new(AtomicBool::new(false));
        let sub_holder: Arc<Mutex<Option<lust::rx::Subscription<i64, String>>>> =
            Arc::new(Mutex::new(None));

        let r_clone = received.clone();
        let sub_clone = sub_holder.clone();
        let sj_clone = self_join_detected.clone();

        let observer = Observer::new(
            move |item: i64| {
                r_clone.lock().unwrap().push(item);
                if item == 2 {
                    if let Some(ref s) = *sub_clone.lock().unwrap() {
                        if let Err(RxError::SelfJoin) = s.join(Some(Duration::from_millis(10))) {
                            sj_clone.store(true, Ordering::SeqCst);
                        }
                        s.cancel();
                    }
                }
                Ok(())
            },
            || {},
            |_: RxError<String>| {},
        );

        let stream: Observable<i64, String> =
            Observable::from_iter(vec![1, 2, 3, 4, 5]);
        let sub = stream
            .subscribe(
                observer,
                scope_id,
                BufferOptions::default_bounded(),
                scheduler.clone(),
            )
            .unwrap();

        *sub_holder.lock().unwrap() = Some(sub.clone());

        scheduler.run_until_quiescent(100);

        assert!(self_join_detected.load(Ordering::SeqCst));
        assert_eq!(*received.lock().unwrap(), vec![1, 2]);

        let completion = sub.join(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(completion.exit, ExitStatus::Cancelled);
    }

    // ------------------------------------------------------------------------
    // Part 5: Virtual Clock Interval & Cancellation
    // ------------------------------------------------------------------------
    {
        let scheduler = Arc::new(RxScheduler::new_virtual());
        let scope_id = 1205;

        let ticks = Arc::new(Mutex::new(Vec::new()));
        let t_clone = ticks.clone();

        let observer: Observer<i64, String> = Observer::from_next(move |val: i64| {
            t_clone.lock().unwrap().push(val);
            Ok(())
        });

        let stream = interval::<String>(Duration::from_millis(100), scheduler.clone());
        let sub = stream
            .subscribe(
                observer,
                scope_id,
                BufferOptions::default_bounded(),
                scheduler.clone(),
            )
            .unwrap();

        scheduler.drain_ready();
        scheduler.advance_and_drain(Duration::from_millis(100)); // tick 0
        scheduler.advance_and_drain(Duration::from_millis(100)); // tick 1
        scheduler.advance_and_drain(Duration::from_millis(100)); // tick 2
        scheduler.advance_and_drain(Duration::from_millis(50));  // 350ms total

        assert_eq!(*ticks.lock().unwrap(), vec![0, 1, 2]);

        sub.cancel();
        scheduler.advance_and_drain(Duration::from_millis(500));
        assert_eq!(*ticks.lock().unwrap(), vec![0, 1, 2]);

        let completion = sub.join(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(completion.exit, ExitStatus::Cancelled);
    }

    // ------------------------------------------------------------------------
    // Part 6: Terminal Error Protocol Idempotence
    // ------------------------------------------------------------------------
    {
        let scheduler = Arc::new(RxScheduler::new_virtual());
        let scope_id = 1206;

        let fail_count = Arc::new(AtomicUsize::new(0));
        let fc = fail_count.clone();

        let observer = Observer::new(
            |_: i64| Ok(()),
            || panic!("Must not complete on error"),
            move |err: RxError<String>| {
                fc.fetch_add(1, Ordering::SeqCst);
                assert_eq!(err, RxError::Source("simulated error".to_string()));
            },
        );

        let stream = Observable::from_producer(|control| {
            control.deliver_next(1)?;
            control.fail(RxError::Source("simulated error".to_string()));
            control.fail(RxError::Source("duplicate error".to_string()));
            control.complete();
            Ok(())
        });

        let sub = stream
            .subscribe(
                observer,
                scope_id,
                BufferOptions::default_bounded(),
                scheduler.clone(),
            )
            .unwrap();

        scheduler.run_until_quiescent(100);

        let completion = sub.join(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(
            completion.exit,
            ExitStatus::Failed(RxError::Source("simulated error".to_string()))
        );
        assert_eq!(fail_count.load(Ordering::SeqCst), 1);
    }
}

// ============================================================================
// Gate-SCHEMA-13: FFI semantic boundary
// ============================================================================
#[test]
fn test_gate_schema_13_ffi_semantic_boundary() {
    // Descriptor and generated-declaration regression checks only. No C/Zig compilation
    // or foreign invocation is performed; semantic FFI safety remains CONCERNS.
    use lust::ast::{FieldOwnership, Span, StructDef, StructField, Type, TypeKind, Visibility};
    use lust::codegen_zig::{generate_c_header, struct_c_abi_descriptor, verify_struct_c_abi_layout, ZigCodegen};

    // Struct definition with nominal layout
    let s = StructDef {
        name: "TelemetryPacket".to_string(),
        type_params: vec![],
        trait_bounds: vec![],
        fields: vec![
            StructField {
                name: "device_id".to_string(),
                ty: Type::new(TypeKind::Int, Span::dummy()),
                visibility: Visibility::Public,
                ownership: FieldOwnership::Strong,
                weak_target: None,
            },
            StructField {
                name: "enabled".to_string(),
                ty: Type::new(TypeKind::Bool, Span::dummy()),
                visibility: Visibility::Public,
                ownership: FieldOwnership::Strong,
                weak_target: None,
            },
            StructField {
                name: "temperature".to_string(),
                ty: Type::new(TypeKind::Float, Span::dummy()),
                visibility: Visibility::Public,
                ownership: FieldOwnership::Strong,
                weak_target: None,
            },
        ],
        visibility: Visibility::Public,
        doc: None,
    };

    // 1. Descriptor identity/order and modeled offset checks
    let desc = struct_c_abi_descriptor(&s);
    assert_eq!(desc.type_name, "TelemetryPacket");
    assert_eq!(desc.fields.len(), 3);
    assert!(verify_struct_c_abi_layout(&s, &desc), "C ABI descriptor must match nominal layout field-for-field");
    assert_eq!(desc.fields[0].name, "device_id");
    assert_eq!(desc.fields[0].byte_offset, 0);
    assert_eq!(desc.fields[1].name, "enabled");
    assert_eq!(desc.fields[1].byte_offset, 8);
    assert_eq!(desc.fields[2].name, "temperature");
    assert_eq!(desc.fields[2].byte_offset, 16);

    // 2. Generated Zig declaration text
    let item = lust::ast::Item::new(lust::ast::ItemKind::Struct(s.clone()), Span::dummy());
    let mut codegen = ZigCodegen::new();
    let zig_out = codegen.generate_program(&[item.clone()]).expect("Zig AOT codegen succeeds");
    assert!(
        zig_out.contains("pub const TelemetryPacket = extern struct {"),
        "Zig AOT struct lowering must use extern struct for nominal C ABI layout"
    );
    assert!(zig_out.contains("device_id: i64,"));
    assert!(zig_out.contains("enabled: bool,"));
    assert!(zig_out.contains("temperature: f64,"));

    // 3. Generated C declaration text
    let c_hdr = generate_c_header(&[item]);
    assert!(c_hdr.contains("struct TelemetryPacket {"));
    assert!(c_hdr.contains("int64_t device_id;"));
    assert!(c_hdr.contains("bool enabled;"));
    assert!(c_hdr.contains("double temperature;"));
}

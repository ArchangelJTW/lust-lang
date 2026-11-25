//! NG-W17-4 (`W16-CRANELIFT-OBJ`): Real Cranelift `ObjectModule` Mach-O / ELF `.o` emission,
//! host linking & native execution parity (VM / Cranelift JIT / Rust AOT / Cranelift AOT),
//! fail-closed target/semantic gates, and isolated child-process `SEC-01` gas watchdog tests.
//!
//! Gate flags: `TASK_ALLOWS_UNSAFE=0` (zero `unsafe` anywhere in this test file).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use lust::codegen_zig::aot::prepare_items_for_mir;
use lust::intern::Interner;
use lust::lexer::Lexer;
use lust::mir::ir::MirModule;
use lust::mir::lowering::AstLowering;
use lust::mir::verifier::MirVerifier;
use lust::parser::Parser;

fn lower_source_to_mir(module_name: &str, source: &str) -> MirModule {
    let mut interner = Interner::new();
    let tokens = Lexer::new(source, &mut interner)
        .tokenize()
        .expect("tokenize Lust source");
    let items = Parser::new(tokens).parse().expect("parse Lust source");
    let prepared = prepare_items_for_mir(&items);
    let mir_module =
        AstLowering::lower_module_items(module_name, &prepared).expect("lower AST to MIR");
    MirVerifier::verify_module(&mir_module).expect("verify lowered MIR module");
    mir_module
}

#[cfg(not(feature = "cranelift-aot"))]
mod feature_off_regression {
    use super::lower_source_to_mir;
    use lust::codegen_zig::aot::{
        CRANELIFT_OBJECT_BACKEND_STATUS, CRANELIFT_OBJECT_BLOCKED_REASON,
        compile_mir_to_object_file, validate_mir_module_clif,
    };
    use tempfile::tempdir;

    #[test]
    fn test_default_build_preserves_fail_closed_cranelift_aot_contract() {
        let source = r#"
function add_mul(a: int, b: int): int
    return a * 3 + b
end
"#;
        let mir_module = lower_source_to_mir("default_off_mod", source);
        let report = validate_mir_module_clif(&mir_module, Some("aarch64-apple-darwin"))
            .expect("CLIF validation succeeds without cranelift-aot");
        assert_eq!(report.object_emission_status, CRANELIFT_OBJECT_BACKEND_STATUS);
        assert!(report.validated_functions.iter().any(|f| f == "add_mul"));

        let tmp = tempdir().expect("tempdir");
        let obj = tmp.path().join("must_not_exist.o");
        let err = compile_mir_to_object_file(&mir_module, &obj, Some("aarch64-apple-darwin"))
            .expect_err("feature-off cranelift-aot must fail closed");
        assert!(
            err.to_string().contains(CRANELIFT_OBJECT_BLOCKED_REASON),
            "unexpected error: {err}"
        );
        assert!(!obj.exists(), "feature-off backend must not write a file");
    }
}

#[cfg(feature = "cranelift-aot")]
mod cranelift_aot_object_tests {
    use super::lower_source_to_mir;
    use cranelift_object::object::{
        Architecture, BinaryFormat, File, Object, ObjectSection, ObjectSymbol, SymbolKind,
    };
    use lust::ast::Span;
    use lust::codegen_zig::aot::cranelift_object_backend::{
        CRANELIFT_OBJECT_BACKEND_STATUS, compile_mir_to_object_file,
        compile_mir_to_object_file_with_report, is_target_isa_compiled, validate_mir_module_clif,
    };
    use lust::codegen_zig::aot::{
        AotCompileOptions, AotTargetKind, RustCompileOptions, ensure_cached_aot_rt_rlib,
    };
    use lust::mir::cranelift_backend::{
        MIR_STATUS_ERR_DIV_BY_ZERO, MIR_STATUS_ERR_OUT_OF_GAS, MIR_STATUS_OK,
    };
    use lust::mir::ir::{
        FunctionSignature, MirConstant, MirFunction, MirModule, MirType, Operand, Place, Rvalue,
        Statement, Terminator,
    };
    use lust::sandbox::{SandboxLimits, SandboxOutcome, SandboxPolicy, run_sandboxed};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::OnceLock;
    use std::time::Duration;
    use tempfile::tempdir;

    fn lust_bin() -> PathBuf {
        PathBuf::from(env!("CARGO_BIN_EXE_lust"))
    }

    fn dylib_filename() -> &'static str {
        if cfg!(target_os = "macos") {
            "liblust.dylib"
        } else if cfg!(target_os = "windows") {
            "lust.dll"
        } else {
            "liblust.so"
        }
    }

    fn locate_or_build_liblust_dir() -> &'static Path {
        static LIB_DIR: OnceLock<PathBuf> = OnceLock::new();
        LIB_DIR
            .get_or_init(|| {
                // `CARGO_BIN_EXE_lust` lives directly inside `<target_dir>/<profile>/`,
                // alongside `liblust.dylib` / `liblust.so` across all Cargo layouts.
                let profile_dir = lust_bin()
                    .parent()
                    .expect("profile dir from CARGO_BIN_EXE_lust")
                    .to_path_buf();
                let target_dir = profile_dir
                    .parent()
                    .expect("target dir from profile dir")
                    .to_path_buf();
                let candidate = profile_dir.join(dylib_filename());
                if candidate.exists() {
                    return profile_dir;
                }
                let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
                let mut cmd = Command::new(&cargo);
                cmd.current_dir(env!("CARGO_MANIFEST_DIR"))
                    .arg("build")
                    .arg("--locked")
                    .arg("--lib")
                    .arg("--features")
                    .arg("cranelift-aot")
                    .arg("--target-dir")
                    .arg(&target_dir);
                if profile_dir.file_name().and_then(|s| s.to_str()) == Some("release") {
                    cmd.arg("--release");
                }
                let out = cmd.output().expect("invoke cargo build --lib for liblust");
                assert!(
                    out.status.success(),
                    "failed to build liblust dylib:\nstdout:\n{}\nstderr:\n{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                assert!(
                    candidate.exists(),
                    "expected {} after cargo build --lib",
                    candidate.display()
                );
                profile_dir
            })
            .as_path()
    }

    fn shared_rust_aot_rlib_cache() -> &'static Path {
        static CACHE: OnceLock<PathBuf> = OnceLock::new();
        CACHE
            .get_or_init(|| {
                let dir = std::env::temp_dir().join("lust-ng-w17-4-rust-aot-rlib-cache");
                let _ = fs::create_dir_all(&dir);
                let opts = RustCompileOptions {
                    aot: AotCompileOptions {
                        target_kind: AotTargetKind::Executable,
                        release_mode: "Debug".to_string(),
                        ..AotCompileOptions::default()
                    },
                    cache_dir: Some(dir.clone()),
                    ..RustCompileOptions::default()
                };
                let _ = ensure_cached_aot_rt_rlib(&opts);
                dir
            })
            .as_path()
    }

    fn link_host_executable_with_liblust(
        obj_path: &Path,
        driver_c_src: &str,
        work_dir: &Path,
        exe_name: &str,
    ) -> PathBuf {
        let driver_c = work_dir.join(format!("{exe_name}_driver.c"));
        let out_exe = work_dir.join(exe_name);
        fs::write(&driver_c, driver_c_src).expect("write C driver source");

        let lib_dir = locate_or_build_liblust_dir();
        let mut cmd = Command::new("cc");
        cmd.arg(obj_path)
            .arg(&driver_c)
            .arg(format!("-L{}", lib_dir.display()))
            .arg("-llust")
            .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
            .arg("-lm")
            .arg("-o")
            .arg(&out_exe);

        let out = cmd.output().expect("invoke host cc linker");
        assert!(
            out.status.success(),
            "host cc link failed for {}:\nstdout:\n{}\nstderr:\n{}",
            exe_name,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        out_exe
    }

    fn run_in_watchdog(exe: &Path, args: &[&str], timeout: Duration) -> SandboxOutcome {
        let mut policy = SandboxPolicy::new(exe);
        policy.args = args.iter().map(|s| (*s).to_string()).collect();
        policy.limits = SandboxLimits {
            wall_timeout: timeout,
            max_stdout_bytes: 256 * 1024,
            max_stderr_bytes: 256 * 1024,
            ..SandboxLimits::default()
        };
        run_sandboxed(&policy)
            .unwrap_or_else(|e| panic!("sandbox supervisor failed for {}: {e}", exe.display()))
    }

    fn normalize_symbol_name(raw: &str) -> &str {
        raw.strip_prefix('_').unwrap_or(raw)
    }

    fn verify_object_has_exported_text_symbols_and_relocs(
        bytes: &[u8],
        expected_format: BinaryFormat,
        expected_arch: Architecture,
        expected_exports: &[&str],
    ) {
        let parsed = File::parse(bytes).expect("object::File::parse must succeed on emitted .o");
        assert_eq!(parsed.format(), expected_format, "unexpected object format");
        assert_eq!(
            parsed.architecture(),
            expected_arch,
            "unexpected object architecture"
        );

        let mut defined_text_syms = Vec::new();
        let mut undefined_syms = Vec::new();
        for sym in parsed.symbols() {
            if let Ok(name) = sym.name() {
                if sym.is_definition() && sym.kind() == SymbolKind::Text {
                    defined_text_syms.push(normalize_symbol_name(name).to_string());
                } else if sym.is_undefined() && !name.is_empty() {
                    undefined_syms.push(normalize_symbol_name(name).to_string());
                }
            }
        }

        for &exp in expected_exports {
            assert!(
                defined_text_syms.iter().any(|s| s == exp),
                "expected defined text symbol '{exp}' in object; found defined={defined_text_syms:?}"
            );
        }

        let mut total_relocs = 0usize;
        for section in parsed.sections() {
            total_relocs += section.relocations().count();
        }
        assert!(
            total_relocs > 0,
            "expected real relocations in emitted object (found 0); undefined={undefined_syms:?}"
        );
    }

    fn deterministic_parity_program_source() -> &'static str {
        r#"
struct Pair
    x: int
    y: int
end

function fib(n: int): int
    if n <= 1 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end

function branch_eval(a: int, b: int): int
    local acc: int = 0
    for i = 1, a do
        if i % 3 == 0 then
            acc = acc + i * 2
        elseif i % 2 == 0 then
            acc = acc + i + b
        else
            acc = acc - i
        end
    end
    return acc
end

function float_accum(n: int): float
    local sum: float = 0.0
    local i: int = 1
    while i <= n do
        sum = sum + 1.5
        i = i + 1
    end
    return sum
end

function struct_total(a: int, b: int): int
    local p: Pair = Pair { x = a, y = b }
    return p.x * 7 + p.y
end

function run_parity(): int
    local f: int = fib(10)
    local b: int = branch_eval(12, 5)
    local fl: float = float_accum(6)
    local st: int = struct_total(4, 9)
    println(f)
    println(b)
    println(fl)
    println(st)
    print(f + b + st, true)
    println(0)
    return f + b + st
end
"#
    }

    #[test]
    fn test_w17_4_host_macho_object_emission_and_inspection() {
        let mir_module = lower_source_to_mir("host_macho_mod", deterministic_parity_program_source());

        let val_report = validate_mir_module_clif(&mir_module, None)
            .expect("validate_mir_module_clif must succeed for host ISA");
        assert_eq!(val_report.object_emission_status, CRANELIFT_OBJECT_BACKEND_STATUS);
        assert!(val_report.validated_functions.iter().any(|f| f == "run_parity"));

        let tmp = tempdir().expect("tempdir");
        let obj_path = tmp.path().join("host_parity.o");
        let report = compile_mir_to_object_file_with_report(&mir_module, &obj_path, None)
            .expect("compile_mir_to_object_file_with_report must emit host .o");
        assert!(obj_path.exists(), "emitted .o file must exist on disk");
        assert!(report.object_size_bytes > 0, "emitted .o must be non-empty");
        assert!(report.exported_functions.iter().any(|f| f == "run_parity"));
        assert!(
            report
                .imported_runtime_symbols
                .iter()
                .any(|s| s == "lust_mir_record_native_entry"),
            "object must import lust_mir_record_native_entry: {:?}",
            report.imported_runtime_symbols
        );

        let bytes = fs::read(&obj_path).expect("read emitted host .o");
        let expected_format = if cfg!(target_os = "macos") {
            BinaryFormat::MachO
        } else {
            BinaryFormat::Elf
        };
        let expected_arch = if cfg!(target_arch = "aarch64") {
            Architecture::Aarch64
        } else if cfg!(target_arch = "x86_64") {
            Architecture::X86_64
        } else {
            panic!("unexpected test host architecture");
        };
        verify_object_has_exported_text_symbols_and_relocs(
            &bytes,
            expected_format,
            expected_arch,
            &["fib", "branch_eval", "float_accum", "struct_total", "run_parity"],
        );

        // Verify host inspection tools (`file`, `nm`, and on macOS `otool -hv`)
        let file_out = Command::new("file").arg(&obj_path).output();
        if let Ok(out) = file_out
            && out.status.success()
        {
            let text = String::from_utf8_lossy(&out.stdout);
            if cfg!(target_os = "macos") {
                assert!(
                    text.contains("Mach-O 64-bit object"),
                    "`file` did not report Mach-O 64-bit object: {text}"
                );
            } else {
                assert!(
                    text.contains("ELF 64-bit"),
                    "`file` did not report ELF 64-bit: {text}"
                );
            }
        }

        let nm_out = Command::new("nm").arg(&obj_path).output();
        if let Ok(out) = nm_out
            && out.status.success()
        {
            let text = String::from_utf8_lossy(&out.stdout);
            assert!(
                text.contains(" T _run_parity") || text.contains(" T run_parity"),
                "`nm` did not report exported text symbol run_parity:\n{text}"
            );
            assert!(
                text.contains(" T _fib") || text.contains(" T fib"),
                "`nm` did not report exported text symbol fib:\n{text}"
            );
        }

        if cfg!(target_os = "macos") {
            let otool_out = Command::new("otool")
                .args(["-hv"])
                .arg(&obj_path)
                .output()
                .expect("run otool -hv on macOS");
            assert!(
                otool_out.status.success(),
                "otool -hv failed: {}",
                String::from_utf8_lossy(&otool_out.stderr)
            );
            let text = String::from_utf8_lossy(&otool_out.stdout);
            assert!(
                text.contains("MH_MAGIC_64") && text.contains("OBJECT"),
                "otool -hv did not confirm MH_MAGIC_64 OBJECT:\n{text}"
            );
        }

        // Also verify CLI `lust build --backend cranelift-aot -o <path.o>` emits a real object file.
        // Include a `main()` entrypoint so `lust build` tree-shaking retains all reachable functions.
        let cli_script = tmp.path().join("cli_emit.lust");
        let cli_obj = tmp.path().join("cli_emit.o");
        let cli_source = format!(
            "{}\nfunction main(): int\n    return run_parity()\nend\n",
            deterministic_parity_program_source()
        );
        fs::write(&cli_script, cli_source).expect("write cli_emit.lust");
        let cli_res = Command::new(lust_bin())
            .arg("build")
            .arg(&cli_script)
            .arg("--backend")
            .arg("cranelift-aot")
            .arg("-o")
            .arg(&cli_obj)
            .output()
            .expect("invoke lust build --backend cranelift-aot");
        assert!(
            cli_res.status.success(),
            "CLI `lust build --backend cranelift-aot` failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&cli_res.stdout),
            String::from_utf8_lossy(&cli_res.stderr)
        );
        let cli_bytes = fs::read(&cli_obj).expect("read CLI-emitted .o");
        verify_object_has_exported_text_symbols_and_relocs(
            &cli_bytes,
            expected_format,
            expected_arch,
            &[
                "cli_emit.fib",
                "cli_emit.branch_eval",
                "cli_emit.float_accum",
                "cli_emit.struct_total",
                "cli_emit.run_parity",
                "cli_emit.main",
            ],
        );
    }

    #[test]
    fn test_w17_4_host_link_execute_and_parity_vs_vm_jit_rust_aot() {
        let base_src = deterministic_parity_program_source();
        let mir_module = lower_source_to_mir("host_parity_exec", base_src);

        let tmp = tempdir().expect("tempdir");
        let obj_path = tmp.path().join("parity_exec.o");
        compile_mir_to_object_file(&mir_module, &obj_path, None)
            .expect("emit host Mach-O .o for parity execution");

        // Link native host executable against liblust with a C driver calling `run_parity()`.
        let driver_c = r#"
#include <stdint.h>
#include <stdio.h>

extern int64_t run_parity(void);
extern int64_t lust_mir_get_status(void);

int main(void) {
    int64_t ret = run_parity();
    int64_t status = lust_mir_get_status();
    if (status != 0) {
        fprintf(stderr, "lust_mir_status=%lld\n", (long long)status);
        return 2;
    }
    printf("RET=%lld\n", (long long)ret);
    fflush(stdout);
    return 0;
}
"#;
        let native_exe =
            link_host_executable_with_liblust(&obj_path, driver_c, tmp.path(), "parity_native");
        let aot_out = run_in_watchdog(&native_exe, &[], Duration::from_secs(10));
        assert!(!aot_out.timed_out, "Cranelift AOT executable timed out");
        assert_eq!(
            aot_out.exit_code,
            Some(0),
            "Cranelift AOT executable failed (stderr: {})",
            String::from_utf8_lossy(&aot_out.stderr)
        );
        let aot_stdout = String::from_utf8_lossy(&aot_out.stdout).into_owned();

        // Build equivalent script that invokes run_parity() and prints the same final line.
        let cli_script_src = format!(
            "{base_src}\nlocal __res: int = run_parity()\nprint(\"RET=\")\nprintln(__res)\n"
        );
        let script_path = tmp.path().join("parity_script.lust");
        fs::write(&script_path, &cli_script_src).expect("write parity_script.lust");

        // 1. Bytecode VM (`--backend vm --jit=off`)
        let vm_out = run_in_watchdog(
            &lust_bin(),
            &[
                "run",
                &script_path.display().to_string(),
                "--backend",
                "vm",
                "--jit=off",
            ],
            Duration::from_secs(15),
        );
        assert!(!vm_out.timed_out, "VM execution timed out");
        assert_eq!(
            vm_out.exit_code,
            Some(0),
            "VM failed: {}",
            String::from_utf8_lossy(&vm_out.stderr)
        );
        let vm_stdout = String::from_utf8_lossy(&vm_out.stdout).into_owned();

        // 2. Cranelift JIT (`--jit=cranelift`)
        let jit_out = run_in_watchdog(
            &lust_bin(),
            &[
                "run",
                &script_path.display().to_string(),
                "--jit=cranelift",
            ],
            Duration::from_secs(15),
        );
        assert!(!jit_out.timed_out, "Cranelift JIT execution timed out");
        assert_eq!(
            jit_out.exit_code,
            Some(0),
            "Cranelift JIT failed: {}",
            String::from_utf8_lossy(&jit_out.stderr)
        );
        let jit_stdout = String::from_utf8_lossy(&jit_out.stdout).into_owned();

        // 3. Rust AOT (`lust build --backend rust --debug`)
        let rust_aot_bin = tmp.path().join("parity_rust_aot");
        let rust_build = Command::new(lust_bin())
            .arg("build")
            .arg(&script_path)
            .arg("--backend")
            .arg("rust")
            .arg("--debug")
            .arg("--cache-dir")
            .arg(shared_rust_aot_rlib_cache())
            .arg("-o")
            .arg(&rust_aot_bin)
            .output()
            .expect("invoke lust build --backend rust");
        assert!(
            rust_build.status.success(),
            "Rust AOT build failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&rust_build.stdout),
            String::from_utf8_lossy(&rust_build.stderr)
        );
        let rust_aot_out = run_in_watchdog(&rust_aot_bin, &[], Duration::from_secs(10));
        assert!(!rust_aot_out.timed_out, "Rust AOT execution timed out");
        assert_eq!(
            rust_aot_out.exit_code,
            Some(0),
            "Rust AOT failed: {}",
            String::from_utf8_lossy(&rust_aot_out.stderr)
        );
        let rust_aot_stdout = String::from_utf8_lossy(&rust_aot_out.stdout).into_owned();

        // Exact 4-way stdout parity (VM == Cranelift JIT == Rust AOT == Cranelift AOT .o)
        assert_eq!(
            aot_stdout, vm_stdout,
            "Cranelift AOT stdout diverged from Bytecode VM"
        );
        assert_eq!(
            aot_stdout, jit_stdout,
            "Cranelift AOT stdout diverged from Cranelift JIT"
        );
        assert_eq!(
            aot_stdout, rust_aot_stdout,
            "Cranelift AOT stdout diverged from Rust AOT"
        );
        assert!(
            aot_stdout.contains("55\n") && aot_stdout.contains("9\n") && aot_stdout.contains("37\n"),
            "unexpected parity output:\n{aot_stdout}"
        );

        // Prove actual native entry via a standalone C runtime shim that counts
        // `lust_mir_record_native_entry` invocations directly from the emitted Mach-O object!
        let native_entry_src = r#"
function add_mul(a: int, b: int): int
    return a * 3 + b
end
"#;
        let ne_mir = lower_source_to_mir("native_entry_mod", native_entry_src);
        let ne_obj = tmp.path().join("native_entry.o");
        compile_mir_to_object_file(&ne_mir, &ne_obj, None).expect("emit native_entry.o");
        let shim_c = r#"
#include <stdint.h>
#include <stdio.h>

extern int64_t add_mul(int64_t a, int64_t b);

static int64_t g_native_entries = 0;
static int64_t g_status = 0;

void lust_mir_record_native_entry(void) { g_native_entries += 1; }
int64_t lust_mir_get_status(void) { return g_status; }
void lust_mir_set_status(int64_t s) { g_status = s; }

int main(void) {
    int64_t v = add_mul(10, 12);
    printf("val=%lld entries=%lld\n", (long long)v, (long long)g_native_entries);
    return (v == 42 && g_native_entries == 1) ? 0 : 1;
}
"#;
        let shim_c_path = tmp.path().join("native_entry_shim.c");
        let shim_exe = tmp.path().join("native_entry_bin");
        fs::write(&shim_c_path, shim_c).expect("write native_entry_shim.c");
        let cc_out = Command::new("cc")
            .arg(&ne_obj)
            .arg(&shim_c_path)
            .arg("-o")
            .arg(&shim_exe)
            .output()
            .expect("link native_entry_bin with standalone C shim");
        assert!(
            cc_out.status.success(),
            "native_entry_bin link failed:\n{}",
            String::from_utf8_lossy(&cc_out.stderr)
        );
        let ne_out = run_in_watchdog(&shim_exe, &[], Duration::from_secs(5));
        assert!(!ne_out.timed_out);
        assert_eq!(ne_out.exit_code, Some(0));
        assert_eq!(
            String::from_utf8_lossy(&ne_out.stdout),
            "val=42 entries=1\n",
            "native entry proof must observe lust_mir_record_native_entry() == 1"
        );

        // Runtime error parity: division by zero sets MIR_STATUS_ERR_DIV_BY_ZERO.
        let div0_src = r#"
function div_zero(a: int, b: int): int
    return a / b
end
"#;
        let div0_mir = lower_source_to_mir("div_zero_mod", div0_src);
        let div0_obj = tmp.path().join("div_zero.o");
        compile_mir_to_object_file(&div0_mir, &div0_obj, None).expect("emit div_zero.o");
        let div0_driver = r#"
#include <stdint.h>
#include <stdio.h>

extern int64_t div_zero(int64_t a, int64_t b);
extern int64_t lust_mir_get_status(void);
extern void lust_mir_set_status(int64_t s);

int main(void) {
    lust_mir_set_status(0);
    int64_t r = div_zero(10, 0);
    int64_t st = lust_mir_get_status();
    printf("ret=%lld status=%lld\n", (long long)r, (long long)st);
    return (st == 0) ? 0 : 10;
}
"#;
        let div0_exe =
            link_host_executable_with_liblust(&div0_obj, div0_driver, tmp.path(), "div0_bin");
        let div0_out = run_in_watchdog(&div0_exe, &[], Duration::from_secs(5));
        assert!(!div0_out.timed_out);
        assert_eq!(div0_out.exit_code, Some(10));
        let div0_stdout = String::from_utf8_lossy(&div0_out.stdout);
        assert!(
            div0_stdout.contains(&format!("status={MIR_STATUS_ERR_DIV_BY_ZERO}")),
            "expected MIR_STATUS_ERR_DIV_BY_ZERO ({MIR_STATUS_ERR_DIV_BY_ZERO}), got: {div0_stdout}"
        );
    }

    #[test]
    fn test_w17_4_elf_object_emission_and_symbol_table_verification() {
        let source = r#"
function helper(x: int): int
    return x * 2 + 1
end

function compute_elf(n: int): int
    local acc: int = 0
    for i = 1, n do
        acc = acc + helper(i)
    end
    return acc
end
"#;
        let mir_module = lower_source_to_mir("elf_target_mod", source);
        let tmp = tempdir().expect("tempdir");

        // 1. Explicit supported ELF target on arm64/all-native-arch: aarch64-unknown-linux-gnu
        if is_target_isa_compiled("aarch64-unknown-linux-gnu") {
            let elf_arm64 = tmp.path().join("compute_aarch64_linux.o");
            let report = compile_mir_to_object_file_with_report(
                &mir_module,
                &elf_arm64,
                Some("aarch64-unknown-linux-gnu"),
            )
            .expect("emit ELF .o for aarch64-unknown-linux-gnu");
            assert_eq!(report.target_triple, "aarch64-unknown-linux-gnu");
            let bytes = fs::read(&elf_arm64).expect("read aarch64 ELF .o");
            assert_eq!(&bytes[..4], b"\x7fELF", "must start with ELF magic header");
            verify_object_has_exported_text_symbols_and_relocs(
                &bytes,
                BinaryFormat::Elf,
                Architecture::Aarch64,
                &["helper", "compute_elf"],
            );

            if let Ok(out) = Command::new("file").arg(&elf_arm64).output()
                && out.status.success()
            {
                let text = String::from_utf8_lossy(&out.stdout);
                assert!(
                    text.contains("ELF 64-bit LSB relocatable"),
                    "unexpected `file` output for aarch64 ELF .o: {text}"
                );
            }
        }

        // 2. x86_64-unknown-linux-gnu: emit & parse ELF if x86 ISA is compiled into
        //    cranelift-codegen; otherwise verify strict fail-closed SupportDisabled error.
        let elf_x86 = tmp.path().join("compute_x86_64_linux.o");
        if is_target_isa_compiled("x86_64-unknown-linux-gnu") {
            let report = compile_mir_to_object_file_with_report(
                &mir_module,
                &elf_x86,
                Some("x86_64-unknown-linux-gnu"),
            )
            .expect("emit ELF .o for x86_64-unknown-linux-gnu");
            assert_eq!(report.target_triple, "x86_64-unknown-linux-gnu");
            let bytes = fs::read(&elf_x86).expect("read x86_64 ELF .o");
            assert_eq!(&bytes[..4], b"\x7fELF", "must start with ELF magic header");
            verify_object_has_exported_text_symbols_and_relocs(
                &bytes,
                BinaryFormat::Elf,
                Architecture::X86_64,
                &["helper", "compute_elf"],
            );
        } else {
            let err = compile_mir_to_object_file(
                &mir_module,
                &elf_x86,
                Some("x86_64-unknown-linux-gnu"),
            )
            .expect_err("disabled x86_64 ISA must fail closed");
            let msg = err.to_string();
            assert!(
                msg.contains("x86_64-unknown-linux-gnu") && msg.contains("disabled"),
                "expected clear disabled-ISA diagnostic for x86_64-unknown-linux-gnu, got: {msg}"
            );
            assert!(
                !elf_x86.exists(),
                "fail-closed target lookup must never leave a partial .o on disk"
            );
        }

        // At least one ELF target must have been emitted and verified on every supported host.
        assert!(
            is_target_isa_compiled("aarch64-unknown-linux-gnu")
                || is_target_isa_compiled("x86_64-unknown-linux-gnu"),
            "host build must compile at least one 64-bit ELF target ISA"
        );
    }

    #[test]
    fn test_w17_4_fail_closed_unsupported_triples_and_unimplemented_semantics() {
        let valid_mir = lower_source_to_mir(
            "valid_mod",
            "function id(x: int): int\n    return x\nend\n",
        );
        let tmp = tempdir().expect("tempdir");

        // 1. Unsupported / malformed / non-object triples fail closed without creating files.
        for bad_triple in ["", "   ", "bogus-unknown-none", "wasm32-unknown-unknown"] {
            let out_path = tmp.path().join("bad_triple.o");
            let err = compile_mir_to_object_file(&valid_mir, &out_path, Some(bad_triple))
                .expect_err(&format!("triple '{bad_triple}' must fail closed"));
            assert!(
                !err.to_string().is_empty(),
                "error diagnostic for '{bad_triple}' must be non-empty"
            );
            assert!(
                !out_path.exists(),
                "must not write file on failed triple '{bad_triple}'"
            );
        }

        // 2. Empty MIR module fails closed.
        let empty_mir = MirModule::new("empty_mod".to_string());
        let empty_obj = tmp.path().join("empty.o");
        assert!(compile_mir_to_object_file(&empty_mir, &empty_obj, None).is_err());
        assert!(!empty_obj.exists());

        // 3. Unknown external callee fails closed.
        let mut unknown_callee_fn = MirFunction::new(
            FunctionSignature {
                name: "calls_unknown".into(),
                param_types: vec![],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let dst = unknown_callee_fn.add_local(Some("r".into()), MirType::Int, false, Span::dummy());
        let ret_bb = unknown_callee_fn.new_block();
        unknown_callee_fn.blocks[0].terminator = Some(Terminator::Call {
            callee: "nonexistent_external_symbol".into(),
            args: vec![],
            destination: Place::Local(dst),
            success: ret_bb,
            error: None,
            span: Span::dummy(),
        });
        unknown_callee_fn.blocks[ret_bb.0].terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(dst))), Span::dummy()));
        let mut unknown_mod = MirModule::new("unknown_callee_mod".into());
        unknown_mod.add_function(unknown_callee_fn);
        let unk_obj = tmp.path().join("unknown_callee.o");
        let err = compile_mir_to_object_file(&unknown_mod, &unk_obj, None)
            .expect_err("unknown external callee must fail closed");
        assert!(err.to_string().contains("nonexistent_external_symbol"));
        assert!(!unk_obj.exists());

        // 4. Algebraic effect/handler intrinsic fails closed in AOT object mode (no VM deopt).
        let mut effect_fn = MirFunction::new(
            FunctionSignature {
                name: "uses_effect".into(),
                param_types: vec![],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let eff_dst = effect_fn.add_local(Some("e".into()), MirType::Int, false, Span::dummy());
        let eff_ret_bb = effect_fn.new_block();
        effect_fn.blocks[0].terminator = Some(Terminator::Call {
            callee: "__lust_perform".into(),
            args: vec![],
            destination: Place::Local(eff_dst),
            success: eff_ret_bb,
            error: None,
            span: Span::dummy(),
        });
        effect_fn.blocks[eff_ret_bb.0].terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(eff_dst))), Span::dummy()));
        let mut eff_mod = MirModule::new("effect_mod".into());
        eff_mod.add_function(effect_fn);
        let eff_obj = tmp.path().join("effect.o");
        let err = compile_mir_to_object_file(&eff_mod, &eff_obj, None)
            .expect_err("VM-only effect intrinsic must fail closed in Cranelift AOT");
        assert!(
            err.to_string().contains("__lust_perform") && err.to_string().contains("not supported"),
            "unexpected effect fail-closed error: {err}"
        );
        assert!(!eff_obj.exists());

        // 5. Static string constant (JIT-process handle) fails closed in AOT object mode.
        let mut str_fn = MirFunction::new(
            FunctionSignature {
                name: "returns_str_const".into(),
                param_types: vec![],
                return_type: MirType::String,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let s_loc = str_fn.add_local(Some("s".into()), MirType::String, false, Span::dummy());
        str_fn.blocks[0].statements.push(Statement::Assign(
            Place::Local(s_loc),
            Rvalue::Use(Operand::Constant(MirConstant::String("hello".into()))),
            Span::dummy(),
        ));
        str_fn.blocks[0].terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(s_loc))), Span::dummy()));
        let mut str_mod = MirModule::new("str_const_mod".into());
        str_mod.add_function(str_fn);
        let str_obj = tmp.path().join("str_const.o");
        compile_mir_to_object_file(&str_mod, &str_obj, None)
            .expect("static string constants should now compile in Cranelift AOT object mode");
        assert!(str_obj.exists());

        // 6. Exported C-ABI runtime helper (`queue.new` -> `lust_mir_queue_new` with `no_mangle`)
        //    now compiles cleanly, while genuinely unexported/unknown intrinsics still fail closed.
        let mut queue_fn = MirFunction::new(
            FunctionSignature {
                name: "uses_queue".into(),
                param_types: vec![],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let q_dst = queue_fn.add_local(Some("q".into()), MirType::Int, false, Span::dummy());
        let q_ret = queue_fn.new_block();
        queue_fn.blocks[0].terminator = Some(Terminator::Call {
            callee: "queue.new".into(),
            args: vec![Operand::Constant(MirConstant::Int(8))],
            destination: Place::Local(q_dst),
            success: q_ret,
            error: None,
            span: Span::dummy(),
        });
        queue_fn.blocks[q_ret.0].terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(q_dst))), Span::dummy()));
        let mut q_mod = MirModule::new("queue_mod".into());
        q_mod.add_function(queue_fn);
        let q_obj = tmp.path().join("queue.o");
        compile_mir_to_object_file(&q_mod, &q_obj, None)
            .expect("exported queue.new runtime symbol should compile in Cranelift AOT");
        assert!(q_obj.exists());

        let mut unexp_fn = MirFunction::new(
            FunctionSignature {
                name: "uses_unexported".into(),
                param_types: vec![],
                return_type: MirType::Int,
                is_public: true,
                effects: None,
            },
            Span::dummy(),
        );
        let u_dst = unexp_fn.add_local(Some("u".into()), MirType::Int, false, Span::dummy());
        let u_ret = unexp_fn.new_block();
        unexp_fn.blocks[0].terminator = Some(Terminator::Call {
            callee: "__lust_unexported_non_c_abi_probe".into(),
            args: vec![],
            destination: Place::Local(u_dst),
            success: u_ret,
            error: None,
            span: Span::dummy(),
        });
        unexp_fn.blocks[u_ret.0].terminator =
            Some(Terminator::Return(Some(Operand::Copy(Place::Local(u_dst))), Span::dummy()));
        let mut u_mod = MirModule::new("unexp_mod".into());
        u_mod.add_function(unexp_fn);
        let u_obj = tmp.path().join("unexported.o");
        let err = compile_mir_to_object_file(&u_mod, &u_obj, None)
            .expect_err("unexported non-C-ABI runtime symbol must fail closed");
        assert!(
            err.to_string().contains("__lust_unexported_non_c_abi_probe"),
            "expected __lust_unexported_non_c_abi_probe fail-closed error, got: {err}"
        );
        assert!(!u_obj.exists());
    }

    #[test]
    fn test_w17_4_sec01_gas_metering_in_isolated_child_with_watchdog() {
        let source = r#"
function bounded_loop(n: int): int
    local count: int = 0
    while count < n do
        count = count + 1
    end
    return count
end

function infinite_loop(): int
    local count: int = 0
    while true do
        count = count + 1
    end
    return count
end
"#;
        let mir_module = lower_source_to_mir("sec01_gas_mod", source);
        let tmp = tempdir().expect("tempdir");
        let obj_path = tmp.path().join("sec01_gas.o");
        compile_mir_to_object_file(&mir_module, &obj_path, None)
            .expect("emit host Mach-O .o for SEC-01 gas metering");

        let driver_c = r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int64_t bounded_loop(int64_t n);
extern int64_t infinite_loop(void);
extern void lust_mir_set_gas(int64_t gas);
extern int64_t lust_mir_get_gas(void);
extern int64_t lust_mir_get_status(void);
extern void lust_mir_set_status(int64_t status);

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: sec01_gas <bounded|infinite> <gas_limit> [n]\n");
        return 64;
    }
    const char *mode = argv[1];
    int64_t gas_limit = strtoll(argv[2], NULL, 10);
    int64_t n = (argc >= 4) ? strtoll(argv[3], NULL, 10) : 0;

    lust_mir_set_status(0);
    lust_mir_set_gas(gas_limit);

    int64_t res = 0;
    if (strcmp(mode, "bounded") == 0) {
        res = bounded_loop(n);
    } else if (strcmp(mode, "infinite") == 0) {
        res = infinite_loop();
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        return 65;
    }

    int64_t status = lust_mir_get_status();
    int64_t gas_rem = lust_mir_get_gas();
    printf("result=%lld status=%lld gas_remaining=%lld\n",
           (long long)res, (long long)status, (long long)gas_rem);
    fflush(stdout);
    return (status == 0) ? 0 : 42;
}
"#;
        let gas_exe =
            link_host_executable_with_liblust(&obj_path, driver_c, tmp.path(), "sec01_gas_bin");
        let watchdog = Duration::from_secs(5);

        // Case 1: Finite sufficient gas -> succeeds, gas decrements monotonically.
        let ok_out = run_in_watchdog(&gas_exe, &["bounded", "1000", "10"], watchdog);
        assert!(
            !ok_out.timed_out,
            "SEC-01 finite gas case was killed by watchdog"
        );
        assert_eq!(ok_out.exit_code, Some(0));
        let ok_stdout = String::from_utf8_lossy(&ok_out.stdout);
        assert!(
            ok_stdout.contains(&format!("result=10 status={MIR_STATUS_OK}")),
            "unexpected finite-gas output: {ok_stdout}"
        );
        let rem_str = ok_stdout
            .split("gas_remaining=")
            .nth(1)
            .expect("gas_remaining field")
            .trim();
        let gas_rem: i64 = rem_str.parse().expect("parse gas_remaining");
        assert!(
            gas_rem < 1000 && gas_rem >= 0,
            "gas_remaining must strictly decrement within [0, 1000), got {gas_rem}"
        );

        // Case 2: Low gas -> terminates cleanly with MIR_STATUS_ERR_OUT_OF_GAS (no watchdog kill).
        let low_out = run_in_watchdog(&gas_exe, &["bounded", "5", "1000"], watchdog);
        assert!(
            !low_out.timed_out,
            "SEC-01 low gas case was killed by watchdog (FAIL)"
        );
        assert_eq!(low_out.exit_code, Some(42));
        let low_stdout = String::from_utf8_lossy(&low_out.stdout);
        assert!(
            low_stdout.contains(&format!("status={MIR_STATUS_ERR_OUT_OF_GAS}")),
            "expected OutOfGas ({MIR_STATUS_ERR_OUT_OF_GAS}) for low gas, got: {low_stdout}"
        );

        // Case 3: Zero gas -> terminates immediately with MIR_STATUS_ERR_OUT_OF_GAS.
        let zero_out = run_in_watchdog(&gas_exe, &["bounded", "0", "10"], watchdog);
        assert!(
            !zero_out.timed_out,
            "SEC-01 zero gas case was killed by watchdog (FAIL)"
        );
        assert_eq!(zero_out.exit_code, Some(42));
        let zero_stdout = String::from_utf8_lossy(&zero_out.stdout);
        assert!(
            zero_stdout.contains(&format!("status={MIR_STATUS_ERR_OUT_OF_GAS}")),
            "expected OutOfGas ({MIR_STATUS_ERR_OUT_OF_GAS}) for zero gas, got: {zero_stdout}"
        );

        // Case 4: Infinite loop with finite gas -> terminates deterministically via OutOfGas.
        let inf_out = run_in_watchdog(&gas_exe, &["infinite", "100"], watchdog);
        assert!(
            !inf_out.timed_out,
            "SEC-01 infinite loop was killed by watchdog instead of halting via gas (FAIL)"
        );
        assert_eq!(inf_out.exit_code, Some(42));
        let inf_stdout = String::from_utf8_lossy(&inf_out.stdout);
        assert!(
            inf_stdout.contains(&format!("status={MIR_STATUS_ERR_OUT_OF_GAS}")),
            "expected OutOfGas ({MIR_STATUS_ERR_OUT_OF_GAS}) for infinite loop, got: {inf_stdout}"
        );
    }
}

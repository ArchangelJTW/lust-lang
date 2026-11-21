#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::sync::Mutex;

use lust::config::LustConfig;
use lust::jit::{
    JitCompiler, Trace, TraceId, is_dynasm_retired, set_dynasm_retired,
};
use lust::{LustError, VM};
use tempfile::tempdir;

static DYNASM_RETIRED_LOCK: Mutex<()> = Mutex::new(());

struct DynasmRetiredGuard {
    prev: bool,
}

impl DynasmRetiredGuard {
    fn set(val: bool) -> Self {
        let prev = is_dynasm_retired();
        set_dynasm_retired(val);
        Self { prev }
    }
}

impl Drop for DynasmRetiredGuard {
    fn drop(&mut self) {
        set_dynasm_retired(self.prev);
    }
}

fn lust_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lust")
}

#[test]
fn test_perf_t1_5_jit_off_disables_dynasm_profiler() {
    let dir = tempdir().expect("create tempdir");
    let script_path = dir.path().join("nested_loop.lust");
    let script = r#"local total: int = 0
local i: int = 0
while i < 60 do
    local j: int = 0
    while j < 60 do
        total = total + i + j
        j = j + 1
    end
    i = i + 1
end
println(total)
"#;
    let mut file = File::create(&script_path).expect("create script");
    file.write_all(script.as_bytes()).expect("write script");

    let modes: &[(&str, &[&str])] = &[
        ("--jit=off", &["--jit=off"]),
        ("--jit=none", &["--jit=none"]),
        ("--jit=vm", &["--jit=vm"]),
        ("--jit=bytecode", &["--jit=bytecode"]),
        ("default", &[]),
    ];

    for (label, extra_args) in modes {
        let output = Command::new(lust_bin())
            .args(*extra_args)
            .arg(&script_path)
            .env("LUST_JIT_STATS", "1")
            .env_remove("LUST_JIT")
            .env_remove("LUST_JIT_BACKEND")
            .env_remove("LUST_BACKEND")
            .output()
            .unwrap_or_else(|e| panic!("failed to run lust ({label}): {e}"));

        assert!(
            output.status.success(),
            "mode {label} failed with status {:?}, stderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("recordings_started: 0"),
            "mode {label} expected recordings_started: 0 in stderr, got:\n{stderr}"
        );
        assert!(
            stderr.contains("recordings_aborted: 0"),
            "mode {label} expected recordings_aborted: 0 in stderr, got:\n{stderr}"
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(
            stdout, "212400\n",
            "mode {label} produced unexpected stdout"
        );
    }
}

#[test]
fn test_perf_t1_5_vm_with_config_disables_jit_when_dynasm_retired() {
    let _lock = DYNASM_RETIRED_LOCK.lock().unwrap();

    // 1. Default retired state (is_dynasm_retired() == true): VM::with_config disables Dynasm JIT.
    assert!(
        is_dynasm_retired(),
        "expected DYNASM_RETIRED to default to true"
    );
    let default_cfg = LustConfig::default();
    let vm_retired = VM::with_config(&default_cfg);
    assert!(
        !vm_retired.jit_configured_enabled(),
        "VM::with_config(&LustConfig::default()) must disable vm.jit.enabled when Dynasm is retired"
    );
    assert!(
        !vm_retired.effective_jit_enabled(),
        "VM::with_config(&LustConfig::default()) must report effective_jit_enabled() == false when Dynasm is retired"
    );

    // 2. When Dynasm retirement is toggled off (RAII-guarded), config flags still govern vm.jit.enabled.
    let _guard = DynasmRetiredGuard::set(false);
    assert!(!is_dynasm_retired());

    let mut cfg_jit_off = LustConfig::default();
    cfg_jit_off.set_jit_enabled(false);
    let vm_jit_off = VM::with_config(&cfg_jit_off);
    assert!(
        !vm_jit_off.jit_configured_enabled(),
        "config.set_jit_enabled(false) must disable JIT even when Dynasm is not retired"
    );
    assert!(!vm_jit_off.effective_jit_enabled());

    let mut cfg_mir = LustConfig::default();
    cfg_mir.set_use_mir(true);
    let vm_mir = VM::with_config(&cfg_mir);
    assert!(
        !vm_mir.jit_configured_enabled(),
        "config.set_use_mir(true) must disable JIT even when Dynasm is not retired"
    );
    assert!(!vm_mir.effective_jit_enabled());

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if !matches!(
        std::env::var("LUST_JIT").as_deref(),
        Ok("0") | Ok("off") | Ok("false")
    ) {
        let vm_active = VM::with_config(&LustConfig::default());
        assert!(
            vm_active.jit_configured_enabled(),
            "when Dynasm is not retired and default config is used, jit_configured_enabled() should be true"
        );
    }
}

#[test]
fn test_perf_t1_5_retired_dynasm_compiler_fallback() {
    let trace = Trace {
        function_idx: 0,
        start_ip: 0,
        is_function: false,
        frame_may_own: false,
        entry_scalars: Vec::new(),
        alias_params: 0,
        borrowed_registers: Vec::new(),
        preamble: Vec::new(),
        ops: Vec::new(),
        postamble: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };

    let mut compiler = JitCompiler::new();
    let err = compiler
        .compile_trace(&trace, TraceId(0), Vec::new())
        .err()
        .expect("JitCompiler::compile_trace must return retired-Dynasm fallback error");

    match err {
        LustError::RuntimeError { message } => {
            assert!(
                message.contains("Dynasm JIT is retired") && message.contains("fallback"),
                "unexpected retired-Dynasm error message: {message}"
            );
        }
        other => panic!("expected LustError::RuntimeError, got: {other:?}"),
    }
}

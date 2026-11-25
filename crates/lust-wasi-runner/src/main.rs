//! Embedded Wasmtime WASI Preview 1 (core module) and Preview 2 (component) runner.
//!
//! Provides a zero-unsafe local WASI execution harness for Wave 17 (`NG-W17-8`)
//! when the standalone `wasmtime` CLI binary is not installed on `PATH`.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use wasmtime::component::{Component, Linker as ComponentLinker, ResourceTable};
use wasmtime::{Config, Engine, Linker, Module, Store};
use wasmtime_wasi::p1::{self, WasiP1Ctx};
use wasmtime_wasi::p2::{self, bindings::sync::Command};
use wasmtime_wasi::{FsPerms, I32Exit, WasiCtx, WasiCtxBuilder, WasiCtxView, WasiView};

struct P2State {
    ctx: WasiCtx,
    table: ResourceTable,
}

impl WasiView for P2State {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.ctx,
            table: &mut self.table,
        }
    }
}

struct CliOpts {
    wasm_path: PathBuf,
    guest_args: Vec<String>,
    dirs: Vec<(String, String, FsPerms)>,
    envs: Vec<(String, String)>,
    inherit_stdio: bool,
    fuel: Option<u64>,
    timeout_ms: Option<u64>,
}

fn parse_dir_spec(spec: &str, perms: FsPerms) -> Result<(String, String, FsPerms), String> {
    if let Some((host, guest)) = spec.split_once("::") {
        Ok((host.to_string(), guest.to_string(), perms))
    } else {
        Ok((spec.to_string(), spec.to_string(), perms))
    }
}

fn parse_args() -> Result<Option<CliOpts>, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() || raw[0] == "--help" || raw[0] == "-h" {
        eprintln!(
            "Usage: lust-wasi-runner [run] [--dir HOST::GUEST] [--dir-ro HOST::GUEST] \
             [--env K=V] [--deny-stdio] [--fuel N] [--timeout-ms MS] <file.wasm> [-- <args...>]"
        );
        return Ok(None);
    }
    if raw[0] == "--version" || raw[0] == "-V" {
        println!("lust-wasi-runner {} (wasmtime 50.0.1)", env!("CARGO_PKG_VERSION"));
        return Ok(None);
    }

    let mut idx = if raw[0] == "run" { 1 } else { 0 };
    let mut dirs = Vec::new();
    let mut envs = Vec::new();
    let mut inherit_stdio = true;
    let mut fuel = None;
    let mut timeout_ms = None;
    let mut wasm_path: Option<PathBuf> = None;
    let mut guest_args: Vec<String> = Vec::new();

    while idx < raw.len() {
        let arg = &raw[idx];
        if wasm_path.is_some() {
            if arg == "--" && guest_args.is_empty() {
                idx += 1;
                continue;
            }
            guest_args.push(arg.clone());
            idx += 1;
            continue;
        }
        match arg.as_str() {
            "--dir" => {
                idx += 1;
                let spec = raw.get(idx).ok_or("--dir requires HOST::GUEST")?;
                dirs.push(parse_dir_spec(spec, FsPerms::ReadWrite)?);
            }
            "--dir-ro" => {
                idx += 1;
                let spec = raw.get(idx).ok_or("--dir-ro requires HOST::GUEST")?;
                dirs.push(parse_dir_spec(spec, FsPerms::ReadOnly)?);
            }
            "--env" => {
                idx += 1;
                let kv = raw.get(idx).ok_or("--env requires K=V")?;
                let (k, v) = kv
                    .split_once('=')
                    .ok_or_else(|| format!("invalid --env '{kv}': expected K=V"))?;
                envs.push((k.to_string(), v.to_string()));
            }
            "--deny-stdio" => {
                inherit_stdio = false;
            }
            "--fuel" => {
                idx += 1;
                let val = raw.get(idx).ok_or("--fuel requires integer")?;
                fuel = Some(
                    val.parse::<u64>()
                        .map_err(|e| format!("invalid --fuel '{val}': {e}"))?,
                );
            }
            "--timeout-ms" => {
                idx += 1;
                let val = raw.get(idx).ok_or("--timeout-ms requires integer")?;
                timeout_ms = Some(
                    val.parse::<u64>()
                        .map_err(|e| format!("invalid --timeout-ms '{val}': {e}"))?,
                );
            }
            "--" => {
                idx += 1;
                if let Some(p) = raw.get(idx) {
                    wasm_path = Some(PathBuf::from(p));
                }
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag: {other}"));
            }
            path => {
                wasm_path = Some(PathBuf::from(path));
            }
        }
        idx += 1;
    }

    let wasm_path = wasm_path.ok_or("missing <file.wasm> argument")?;
    Ok(Some(CliOpts {
        wasm_path,
        guest_args,
        dirs,
        envs,
        inherit_stdio,
        fuel,
        timeout_ms,
    }))
}

fn configure_wasi(opts: &CliOpts) -> Result<WasiCtxBuilder, String> {
    let mut builder = WasiCtxBuilder::new();
    if opts.inherit_stdio {
        builder.inherit_stdio();
    }
    let mut full_args = Vec::with_capacity(opts.guest_args.len() + 1);
    full_args.push(
        opts.wasm_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("guest.wasm")
            .to_string(),
    );
    full_args.extend(opts.guest_args.iter().cloned());
    builder.args(&full_args);
    for (k, v) in &opts.envs {
        builder.env(k, v);
    }
    for (host, guest, perms) in &opts.dirs {
        builder
            .preopened_dir(host, guest, *perms)
            .map_err(|e| format!("preopened_dir({host}::{guest}) failed: {e}"))?;
    }
    Ok(builder)
}

fn is_wasm_component(bytes: &[u8]) -> bool {
    // WebAssembly Component binary header: \0asm + version 0x0d 0x00 0x01 0x00 (layer 1)
    bytes.len() >= 8 && &bytes[0..4] == b"\0asm" && bytes[4..8] == [0x0d, 0x00, 0x01, 0x00]
}

fn run(opts: CliOpts) -> Result<i32, String> {
    let bytes = std::fs::read(&opts.wasm_path)
        .map_err(|e| format!("failed reading '{}': {e}", opts.wasm_path.display()))?;

    let mut config = Config::new();
    if opts.fuel.is_some() {
        config.consume_fuel(true);
    }
    if opts.timeout_ms.is_some() {
        config.epoch_interruption(true);
    }

    let engine = Engine::new(&config).map_err(|e| format!("engine init failed: {e}"))?;

    let mut builder = configure_wasi(&opts)?;

    let spawn_timeout_watchdog = |timeout_ms: Option<u64>, engine: &Engine| -> Option<Arc<AtomicBool>> {
        let ms = timeout_ms?;
        let done = Arc::new(AtomicBool::new(false));
        let done_flag = Arc::clone(&done);
        let engine_clone = engine.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(ms));
            while !done_flag.load(Ordering::Relaxed) {
                engine_clone.increment_epoch();
                std::thread::sleep(Duration::from_millis(10));
            }
        });
        Some(done)
    };

    if is_wasm_component(&bytes) {
        let component = Component::from_binary(&engine, &bytes)
            .map_err(|e| format!("component compile failed: {e}"))?;
        let mut linker: ComponentLinker<P2State> = ComponentLinker::new(&engine);
        p2::add_to_linker_sync(&mut linker)
            .map_err(|e| format!("wasi p2 linker failed: {e}"))?;
        let mut store = Store::new(
            &engine,
            P2State {
                ctx: builder.build(),
                table: ResourceTable::new(),
            },
        );
        if let Some(f) = opts.fuel {
            store
                .set_fuel(f)
                .map_err(|e| format!("set_fuel failed: {e}"))?;
        }
        if opts.timeout_ms.is_some() {
            store.set_epoch_deadline(1);
        }
        let command = Command::instantiate(&mut store, &component, &linker)
            .map_err(|e| format!("component instantiate failed: {e}"))?;
        let done = spawn_timeout_watchdog(opts.timeout_ms, &engine);
        let res = command.wasi_cli_run().call_run(&mut store);
        if let Some(flag) = done {
            flag.store(true, Ordering::Relaxed);
        }
        match res {
            Ok(Ok(())) => Ok(0),
            Ok(Err(())) => Ok(1),
            Err(err) => {
                if let Some(exit) = err.downcast_ref::<I32Exit>() {
                    Ok(exit.0)
                } else {
                    Err(format!("component execution trapped: {err}"))
                }
            }
        }
    } else {
        let module = Module::from_binary(&engine, &bytes)
            .map_err(|e| format!("module compile failed: {e}"))?;
        let mut linker: Linker<WasiP1Ctx> = Linker::new(&engine);
        p1::add_to_linker_sync(&mut linker, |t| t)
            .map_err(|e| format!("wasi p1 linker failed: {e}"))?;
        let mut store = Store::new(&engine, builder.build_p1());
        if let Some(f) = opts.fuel {
            store
                .set_fuel(f)
                .map_err(|e| format!("set_fuel failed: {e}"))?;
        }
        if opts.timeout_ms.is_some() {
            store.set_epoch_deadline(1);
        }
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| format!("module instantiate failed: {e}"))?;
        let start = instance
            .get_typed_func::<(), ()>(&mut store, "_start")
            .map_err(|e| format!("missing _start export: {e}"))?;
        let done = spawn_timeout_watchdog(opts.timeout_ms, &engine);
        let res = start.call(&mut store, ());
        if let Some(flag) = done {
            flag.store(true, Ordering::Relaxed);
        }
        match res {
            Ok(()) => Ok(0),
            Err(err) => {
                if let Some(exit) = err.downcast_ref::<I32Exit>() {
                    Ok(exit.0)
                } else {
                    Err(format!("module execution trapped: {err}"))
                }
            }
        }
    }
}

fn main() -> ExitCode {
    let opts = match parse_args() {
        Ok(Some(o)) => o,
        Ok(None) => return ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("lust-wasi-runner error: {e}");
            return ExitCode::from(2);
        }
    };
    match run(opts) {
        Ok(code) => ExitCode::from((code & 0xff) as u8),
        Err(e) => {
            eprintln!("lust-wasi-runner error: {e}");
            ExitCode::from(1)
        }
    }
}

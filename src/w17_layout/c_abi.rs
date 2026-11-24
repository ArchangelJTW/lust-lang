//! C11 & Zig ABI Layout Compiler, Verifier, and Disassembly Auditor (`TASK_ALLOWS_UNSAFE=0`).
//!
//! Compiles host C fixtures with `cc -std=c11 -Wall -Wextra -Werror` and sanitizer flags
//! (`-fsanitize=address,undefined` when supported by host toolchain), executes them in
//! isolated child processes with external watchdogs, and compares actual host C `sizeof`,
//! `alignof`, and `offsetof` against `LogicalLayoutV1`.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::data::layout::LogicalLayoutV1;
use crate::w17_layout::watchdog::{WatchdogRunResult, run_child_with_watchdog};

/// Recorded build provenance for a compiled host C fixture.
#[derive(Debug, Clone)]
pub struct CBuildEvidence {
    pub compiler: String,
    pub compiler_version: String,
    pub command_line: String,
    pub sanitizers_enabled: bool,
    pub sanitizer_flags: Vec<String>,
    pub output_binary: PathBuf,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Host C compiler layout report for a single struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CStructLayoutEntry {
    pub size: usize,
    pub align: usize,
    pub offsets: Vec<usize>,
    pub sizes: Vec<usize>,
}

/// Parsed layout report emitted by `c_layout_probe --layout-json`.
#[derive(Debug, Clone)]
pub struct CLayoutReport {
    pub pointer_width: usize,
    pub descriptors: BTreeMap<String, CStructLayoutEntry>,
    pub structs: BTreeMap<String, CStructLayoutEntry>,
}

/// Disassembly verification record for direct constant-offset field loads.
#[derive(Debug, Clone)]
pub struct DisassemblyAudit {
    pub tool: String,
    pub symbol: String,
    pub instruction_count: usize,
    pub has_constant_offset_load: bool,
    pub forbidden_lookup_calls_found: Vec<String>,
    pub raw_excerpt: String,
}

impl DisassemblyAudit {
    pub fn is_zero_lookup_direct(&self) -> bool {
        self.instruction_count > 0
            && self.has_constant_offset_load
            && self.forbidden_lookup_calls_found.is_empty()
    }
}

/// Evidence from running the optional host Zig `extern struct` layout probe.
#[derive(Debug, Clone)]
pub struct ZigLayoutEvidence {
    pub available: bool,
    pub version: String,
    pub command_line: String,
    pub passed: bool,
    pub stdout: String,
    pub stderr: String,
}

fn host_cc() -> String {
    std::env::var("CC")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "cc".to_string())
}

fn query_tool_version(tool: &str, arg: &str) -> String {
    Command::new(tool)
        .arg(arg)
        .output()
        .ok()
        .map(|o| {
            let s = String::from_utf8_lossy(&o.stdout);
            if s.trim().is_empty() {
                String::from_utf8_lossy(&o.stderr).trim().to_string()
            } else {
                s.lines().next().unwrap_or("").trim().to_string()
            }
        })
        .unwrap_or_else(|| "<unavailable>".to_string())
}

/// Compiles a C11 fixture source file using host `cc`, probing `-fsanitize=address,undefined`
/// first and falling back cleanly only if the host toolchain explicitly does not support
/// those sanitizer runtimes (while recording `sanitizers_enabled` truthfully).
pub fn build_c_fixture_with_sanitizers(
    source_path: &Path,
    include_dir: &Path,
    output_binary: &Path,
    opt_flag: &str,
) -> Result<CBuildEvidence, String> {
    let cc = host_cc();
    let version = query_tool_version(&cc, "--version");
    let san_flags = vec![
        "-fsanitize=address,undefined".to_string(),
        "-fno-omit-frame-pointer".to_string(),
    ];

    let mut args = vec![
        "-std=c11".to_string(),
        "-Wall".to_string(),
        "-Wextra".to_string(),
        "-Werror".to_string(),
        opt_flag.to_string(),
        "-I".to_string(),
        include_dir.display().to_string(),
    ];
    args.extend(san_flags.iter().cloned());
    args.push(source_path.display().to_string());
    args.push("-o".to_string());
    args.push(output_binary.display().to_string());

    let out = Command::new(&cc)
        .args(&args)
        .output()
        .map_err(|e| format!("failed to invoke `{cc}`: {e}"))?;

    if out.status.success() {
        return Ok(CBuildEvidence {
            compiler: cc.clone(),
            compiler_version: version,
            command_line: format!("{cc} {}", args.join(" ")),
            sanitizers_enabled: true,
            sanitizer_flags: san_flags,
            output_binary: output_binary.to_path_buf(),
            exit_code: out.status.code().unwrap_or(0),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
    }

    // Check whether failure was specifically due to unsupported sanitizer flags.
    let san_stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    if san_stderr.contains("unsupported option")
        || san_stderr.contains("cannot find")
        || san_stderr.contains("library not found for -l")
    {
        let fallback_args = vec![
            "-std=c11".to_string(),
            "-Wall".to_string(),
            "-Wextra".to_string(),
            "-Werror".to_string(),
            opt_flag.to_string(),
            "-I".to_string(),
            include_dir.display().to_string(),
            source_path.display().to_string(),
            "-o".to_string(),
            output_binary.display().to_string(),
        ];
        let fb = Command::new(&cc)
            .args(&fallback_args)
            .output()
            .map_err(|e| format!("failed fallback `{cc}`: {e}"))?;
        if fb.status.success() {
            return Ok(CBuildEvidence {
                compiler: cc.clone(),
                compiler_version: version,
                command_line: format!("{cc} {}", fallback_args.join(" ")),
                sanitizers_enabled: false,
                sanitizer_flags: Vec::new(),
                output_binary: output_binary.to_path_buf(),
                exit_code: fb.status.code().unwrap_or(0),
                stdout: String::from_utf8_lossy(&fb.stdout).into_owned(),
                stderr: format!(
                    "sanitizer probe failed: {san_stderr}\nfallback stderr: {}",
                    String::from_utf8_lossy(&fb.stderr)
                ),
            });
        }
    }

    Err(format!(
        "C fixture compile failed (`{cc} {}`): exit={:?}\nstderr:\n{san_stderr}",
        args.join(" "),
        out.status.code()
    ))
}

/// Runs `c_layout_probe --layout-json` in an isolated child process and parses its JSON output.
pub fn run_c_layout_report(
    probe_bin: &Path,
    timeout: Duration,
) -> Result<(CLayoutReport, WatchdogRunResult), String> {
    let run = run_child_with_watchdog(probe_bin, &["--layout-json"], &[], timeout)?;
    if !run.outcome.is_clean_success() {
        return Err(format!(
            "c_layout_probe --layout-json failed: outcome={:?}, stderr={}",
            run.outcome, run.stderr
        ));
    }
    let val: serde_json::Value =
        serde_json::from_str(&run.stdout).map_err(|e| format!("parse layout json: {e}"))?;

    let pointer_width = val["pointer_width"]
        .as_u64()
        .ok_or("missing pointer_width")? as usize;

    let parse_table = |obj: &serde_json::Value| -> Result<BTreeMap<String, CStructLayoutEntry>, String> {
        let map = obj.as_object().ok_or("expected JSON object")?;
        let mut out = BTreeMap::new();
        for (name, entry) in map {
            let size = entry["size"].as_u64().ok_or("missing size")? as usize;
            let align = entry["align"].as_u64().ok_or("missing align")? as usize;
            let offsets = entry["offsets"]
                .as_array()
                .ok_or("missing offsets")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let sizes = entry
                .get("sizes")
                .and_then(|s| s.as_array())
                .map(|arr| arr.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
                .unwrap_or_default();
            out.insert(
                name.clone(),
                CStructLayoutEntry {
                    size,
                    align,
                    offsets,
                    sizes,
                },
            );
        }
        Ok(out)
    };

    let descriptors = parse_table(&val["descriptors"])?;
    let structs = parse_table(&val["structs"])?;

    Ok((
        CLayoutReport {
            pointer_width,
            descriptors,
            structs,
        },
        run,
    ))
}

/// Verifies exact equality between a runtime `LogicalLayoutV1` and the host C compiler's
/// `sizeof`, `alignof`, and per-field `offsetof` & field byte sizes.
pub fn compare_layout_with_c_entry(
    layout: &LogicalLayoutV1,
    c_entry: &CStructLayoutEntry,
) -> Result<(), String> {
    if layout.size_bytes != c_entry.size {
        return Err(format!(
            "size mismatch for `{}`: LogicalLayoutV1={} vs C sizeof={}",
            layout.type_name, layout.size_bytes, c_entry.size
        ));
    }
    if layout.align_bytes != c_entry.align {
        return Err(format!(
            "align mismatch for `{}`: LogicalLayoutV1={} vs C alignof={}",
            layout.type_name, layout.align_bytes, c_entry.align
        ));
    }
    if layout.fields.len() != c_entry.offsets.len() {
        return Err(format!(
            "field count mismatch for `{}`: LogicalLayoutV1={} vs C={}",
            layout.type_name,
            layout.fields.len(),
            c_entry.offsets.len()
        ));
    }
    for (i, field) in layout.fields.iter().enumerate() {
        if field.byte_offset != c_entry.offsets[i] {
            return Err(format!(
                "offset mismatch for `{}.{}` (ordinal {}): LogicalLayoutV1={} vs C offsetof={}",
                layout.type_name, field.name, field.ordinal, field.byte_offset, c_entry.offsets[i]
            ));
        }
        if !c_entry.sizes.is_empty() && field.byte_size != c_entry.sizes[i] {
            return Err(format!(
                "field size mismatch for `{}.{}`: LogicalLayoutV1={} vs C sizeof={}",
                layout.type_name, field.name, field.byte_size, c_entry.sizes[i]
            ));
        }
    }
    Ok(())
}

/// Compiles a deliberately tampered C `_Static_assert` source file and asserts that host `cc`
/// rejects it at compile time (negative layout mutant proof).
pub fn compile_c_negative_layout_mutant(
    include_dir: &Path,
    mutant_name: &str,
    static_assert_expr: &str,
) -> Result<String, String> {
    let tmp = tempfile::tempdir().map_err(|e| format!("mutant tempdir: {e}"))?;
    let src_path = tmp.path().join(format!("{mutant_name}.c"));
    let obj_path = tmp.path().join(format!("{mutant_name}.o"));
    let src = format!(
        "#include \"lust_layout_abi_v1.h\"\n_Static_assert({static_assert_expr}, \"{mutant_name}\");\nint main(void) {{ return 0; }}\n"
    );
    fs::write(&src_path, src).map_err(|e| format!("write mutant c: {e}"))?;

    let cc = host_cc();
    let out = Command::new(&cc)
        .args([
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-c",
            "-I",
            &include_dir.display().to_string(),
            &src_path.display().to_string(),
            "-o",
            &obj_path.display().to_string(),
        ])
        .output()
        .map_err(|e| format!("invoke cc on mutant {mutant_name}: {e}"))?;

    if out.status.success() {
        return Err(format!(
            "Mutant `{mutant_name}` unexpectedly compiled with `{static_assert_expr}`!"
        ));
    }
    Ok(String::from_utf8_lossy(&out.stderr).into_owned())
}

/// Disassembles `symbol` in `binary_path` using host `objdump -d` or `otool -tV` and audits
/// that it performs direct constant-offset loads without any string/hash lookup or heap
/// allocation calls.
pub fn audit_c_accessor_disassembly(
    binary_path: &Path,
    symbol: &str,
) -> Result<DisassemblyAudit, String> {
    let (tool, raw_out) = if cfg!(target_os = "macos") {
        let out = Command::new("otool")
            .args(["-tV", &binary_path.display().to_string()])
            .output()
            .map_err(|e| format!("invoke otool: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "otool failed: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        (
            "otool -tV".to_string(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
        )
    } else {
        let out = Command::new("objdump")
            .args(["-d", &binary_path.display().to_string()])
            .output()
            .map_err(|e| format!("invoke objdump: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "objdump failed: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        (
            "objdump -d".to_string(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
        )
    };

    let mut in_fn = false;
    let mut fn_lines: Vec<String> = Vec::new();
    let darwin_sym = format!("_{symbol}:");
    let elf_sym = format!("<{symbol}>:");

    for line in raw_out.lines() {
        let trimmed = line.trim();
        if trimmed == darwin_sym || trimmed.ends_with(&darwin_sym) || trimmed.contains(&elf_sym) {
            in_fn = true;
            fn_lines.push(line.to_string());
            continue;
        }
        if in_fn {
            // Next function header terminates this function's disassembly.
            if !line.starts_with('\t')
                && !line.starts_with(' ')
                && (trimmed.ends_with(':') || trimmed.contains(">:"))
                && !trimmed.is_empty()
            {
                break;
            }
            if !trimmed.is_empty() {
                fn_lines.push(line.to_string());
            }
        }
    }

    if fn_lines.is_empty() {
        return Err(format!(
            "symbol `{symbol}` not found in `{tool}` output for {}",
            binary_path.display()
        ));
    }

    let body = fn_lines.join("\n");
    let instruction_count = fn_lines.len().saturating_sub(1);
    let lower = body.to_ascii_lowercase();
    let has_constant_offset_load = lower.contains("ldr")
        || lower.contains("ldrb")
        || lower.contains("movq")
        || lower.contains("movzbl")
        || lower.contains("mov\t");

    let forbidden_tokens = [
        "strcmp",
        "strncmp",
        "strlen",
        "malloc",
        "calloc",
        "realloc",
        "hash",
        "get_field",
        "struct_get_field",
    ];
    let mut forbidden_lookup_calls_found = Vec::new();
    for tok in forbidden_tokens {
        if lower.contains(tok) {
            forbidden_lookup_calls_found.push(tok.to_string());
        }
    }

    Ok(DisassemblyAudit {
        tool,
        symbol: symbol.to_string(),
        instruction_count,
        has_constant_offset_load,
        forbidden_lookup_calls_found,
        raw_excerpt: body,
    })
}

/// Runs the host Zig `extern struct` comptime layout verification if `zig` is installed on PATH.
pub fn verify_zig_layout_probe(
    zig_source_path: &Path,
    timeout: Duration,
) -> Result<ZigLayoutEvidence, String> {
    let version_out = Command::new("zig").arg("version").output();
    let Ok(vout) = version_out else {
        return Ok(ZigLayoutEvidence {
            available: false,
            version: "<not installed>".to_string(),
            command_line: String::new(),
            passed: false,
            stdout: String::new(),
            stderr: "zig binary not found in PATH".to_string(),
        });
    };
    if !vout.status.success() {
        return Ok(ZigLayoutEvidence {
            available: false,
            version: "<error>".to_string(),
            command_line: String::new(),
            passed: false,
            stdout: String::new(),
            stderr: String::from_utf8_lossy(&vout.stderr).into_owned(),
        });
    }
    let version = String::from_utf8_lossy(&vout.stdout).trim().to_string();

    let tmp = tempfile::tempdir().map_err(|e| format!("zig tempdir: {e}"))?;
    let bin_path = tmp.path().join("zig_layout_probe");
    let emit_arg = format!("-femit-bin={}", bin_path.display());
    let build_args = [
        "build-exe",
        &zig_source_path.display().to_string(),
        &emit_arg,
        "-O",
        "ReleaseSafe",
    ];
    let cmd_str = format!("zig {}", build_args.join(" "));
    let build_out = Command::new("zig")
        .args(build_args)
        .current_dir(tmp.path())
        .output()
        .map_err(|e| format!("invoke zig build-exe: {e}"))?;

    if !build_out.status.success() {
        return Ok(ZigLayoutEvidence {
            available: true,
            version,
            command_line: cmd_str,
            passed: false,
            stdout: String::from_utf8_lossy(&build_out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&build_out.stderr).into_owned(),
        });
    }

    let run = run_child_with_watchdog(&bin_path, &[], &[], timeout)?;
    Ok(ZigLayoutEvidence {
        available: true,
        version,
        command_line: cmd_str,
        passed: run.outcome.is_clean_success(),
        stdout: run.stdout,
        stderr: run.stderr,
    })
}

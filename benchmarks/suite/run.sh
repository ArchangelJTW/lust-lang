#!/bin/bash
# Cross-language suite: Native Rust, Lust AOT (--backend rust|zig|mir|mir-zig),
# LuaJIT, Lust Cranelift JIT (--jit=cranelift), Lust Bytecode VM (--jit=off), Lua.
#
# Usage:
#   benchmarks/suite/run.sh [path/to/lust] [--backend rust|zig|mir|mir-zig] [--cache-dir <dir>]
#
# Defaults:
#   LUST        = target/release/lust
#   AOT_BACKEND = rust (override with --backend or LUST_AOT_BACKEND)
#   TIMEOUT     = 120 seconds per program
#
# Each engine gets at most $TIMEOUT seconds per program, so one slow row reports
# "timeout" instead of hanging the suite. Every program is run in a private
# working directory so no stray file can perturb a later run.
set -euo pipefail

D=$(cd "$(dirname "$0")" && pwd)
LUST=""
AOT_BACKEND="${LUST_AOT_BACKEND:-rust}"
AOT_CACHE_DIR="${LUST_AOT_CACHE_DIR:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            if [[ $# -lt 2 ]]; then
                echo "Error: --backend requires an argument (rust, zig, mir, mir-zig)" >&2
                exit 2
            fi
            AOT_BACKEND="$2"
            shift 2
            ;;
        --backend=*)
            AOT_BACKEND="${1#--backend=}"
            shift
            ;;
        --cache-dir)
            if [[ $# -lt 2 ]]; then
                echo "Error: --cache-dir requires a directory argument" >&2
                exit 2
            fi
            AOT_CACHE_DIR="$2"
            shift 2
            ;;
        --cache-dir=*)
            AOT_CACHE_DIR="${1#--cache-dir=}"
            shift
            ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            if [[ -z "$LUST" ]]; then
                LUST="$1"
                shift
            else
                echo "Error: unexpected argument '$1'" >&2
                exit 2
            fi
            ;;
    esac
done

LUST="${LUST:-$D/../../target/release/lust}"
RUST_MANIFEST="$D/rust/Cargo.toml"
RUST_BIN="$D/rust/target/release/suite-rust"
TIMEOUT="${TIMEOUT:-120}"

cargo build --release --manifest-path "$RUST_MANIFEST" >&2

export D LUST AOT_BACKEND AOT_CACHE_DIR RUST_BIN TIMEOUT

exec python3 - << 'EOF'
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

suite_dir = Path(os.environ["D"])
lust_bin = Path(os.environ["LUST"])
rust_bin = Path(os.environ["RUST_BIN"])
aot_backend = os.environ.get("AOT_BACKEND", "rust")
explicit_cache_dir = os.environ.get("AOT_CACHE_DIR", "").strip()
timeout_s = int(os.environ.get("TIMEOUT", "120"))

has_luajit = shutil.which("luajit") is not None
has_lua = shutil.which("lua") is not None

programs = [
    "fields",
    "array",
    "sieve",
    "calls",
    "methods",
    "strings",
    "fib",
    "nested",
    "floatmath",
    "tree",
]

with tempfile.TemporaryDirectory(dir="/tmp") as td:
    td_path = Path(td)
    rlib_cache_dir = Path(explicit_cache_dir) if explicit_cache_dir else (td_path / "rlib_cache")

    def run_cmd(cmd, env=None):
        e = os.environ.copy()
        if env:
            e.update(env)
        start = time.perf_counter_ns()
        try:
            res = subprocess.run(
                cmd,
                cwd=td,
                env=e,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            dur_ms = (time.perf_counter_ns() - start) / 1_000_000
            last_line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
            if res.returncode != 0:
                return None, "crash/error", f"<exit {res.returncode}>"
            return dur_ms, f"{dur_ms:6.1f} ms", last_line
        except subprocess.TimeoutExpired:
            return None, "timeout", "<timeout>"
        except FileNotFoundError:
            return None, "n/a", ""

    aot_bins = {}
    print(
        f"Building Lust AOT binaries (--backend {aot_backend} --release-fast)...",
        file=sys.stderr,
    )
    for prog in programs:
        bin_path = td_path / f"{prog}_aot"
        build_cmd = [
            str(lust_bin),
            "build",
            str(suite_dir / f"{prog}.lust"),
            "--backend",
            aot_backend,
            "-o",
            str(bin_path),
            "--release-fast",
        ]
        if aot_backend.lower() == "rust":
            build_cmd.extend(["--cache-dir", str(rlib_cache_dir)])
        res = subprocess.run(build_cmd, capture_output=True, text=True)
        if res.returncode == 0 and bin_path.exists():
            aot_bins[prog] = bin_path
            # One warmup execution to prime dyld / code signing cache
            subprocess.run([str(bin_path)], capture_output=True)
        else:
            print(f"Failed to build AOT ({aot_backend}) for {prog}: {res.stderr}", file=sys.stderr)

    print(
        f"{'program':<10} {'rust':>9} {'lust-aot':>10} {'aot/rust':>8} {'luajit':>9} {'lust-jit':>10} {'lust-vm':>10} {'lua':>9}   outputs"
    )

    had_lust_mismatch = False
    for prog in programs:
        ms_rust, t_rust, o_rust = run_cmd([str(rust_bin), prog])
        if prog in aot_bins:
            ms_aot, t_aot, o_aot = run_cmd([str(aot_bins[prog])])
        else:
            ms_aot, t_aot, o_aot = None, "n/a", ""

        if ms_rust and ms_aot is not None and ms_rust > 0:
            ratio_str = f"{ms_aot / ms_rust:7.2f}x"
        else:
            ratio_str = "     n/a"

        _, t_jit, o_jit = run_cmd(
            [str(lust_bin), "run", str(suite_dir / f"{prog}.lust"), "--jit=cranelift"]
        )
        _, t_vm, o_vm = run_cmd(
            [str(lust_bin), "run", str(suite_dir / f"{prog}.lust"), "--jit=off"]
        )
        if has_luajit:
            _, t_lj, o_lj = run_cmd(["luajit", str(suite_dir / f"{prog}.lua")])
        else:
            t_lj, o_lj = "n/a", ""
        if has_lua:
            _, t_lua, o_lua = run_cmd(["lua", str(suite_dir / f"{prog}.lua")])
        else:
            t_lua, o_lua = "n/a", ""

        lust_outputs = [("aot", o_aot), ("jit", o_jit), ("vm", o_vm)]
        ext_outputs = []
        if has_luajit:
            ext_outputs.append(("luajit", o_lj))
        if has_lua:
            ext_outputs.append(("lua", o_lua))

        def match_val(a, b, exact_only=False):
            if a == b:
                return True
            if exact_only:
                return False
            try:
                fa, fb = float(a), float(b)
                return math.isclose(fa, fb, rel_tol=1e-6)
            except ValueError:
                return False

        lust_diff = [name for name, output in lust_outputs if not match_val(o_rust, output, exact_only=True)]
        ext_diff = [name for name, output in ext_outputs if not match_val(o_rust, output, exact_only=False)]
        diff = lust_diff + ext_diff
        if lust_diff:
            had_lust_mismatch = True

        status = "same" if not diff else "DIFF: rust!=" + ",".join(diff)
        print(
            f"{prog:<10} {t_rust:>9} {t_aot:>10} {ratio_str:>8} {t_lj:>9} {t_jit:>10} {t_vm:>10} {t_lua:>9}   {status}"
        )

    if had_lust_mismatch:
        sys.exit(1)
EOF

#!/bin/bash
# Cross-language suite: Lust interpreter, Lust JIT, LuaJIT, Lua.
# Usage: benchmarks/suite/run.sh [path/to/lust]   (default: target/release/lust)
#
# Each engine gets at most $TIMEOUT seconds per program (default 120), so one
# slow row reports "timeout" instead of hanging the suite. Every program is
# run in a private working directory: the suite writes no files, but that
# keeps a stray one from changing a later run.
D=$(cd "$(dirname "$0")" && pwd)
LUST=${1:-"$D/../../target/release/lust"}
RUST_MANIFEST="$D/rust/Cargo.toml"
RUST_BIN="$D/rust/target/release/suite-rust"
cargo build --release --manifest-path "$RUST_MANIFEST" >&2
TIMEOUT=${TIMEOUT:-120}
exec python3 - << 'EOF'
import os, sys, time, subprocess, tempfile
from pathlib import Path

suite_dir = Path(os.environ.get("D", "/Users/vedmalex/work/lust-lang/repos/lust-lang-fork/benchmarks/suite"))
lust_bin = Path(sys.argv[1]) if len(sys.argv) > 1 else suite_dir.parent.parent / "target/release/lust"
rust_bin = suite_dir / "rust/target/release/suite-rust"

programs = ["fields", "array", "sieve", "calls", "methods", "strings", "fib", "nested", "floatmath", "tree"]

with tempfile.TemporaryDirectory(dir="/tmp") as td:
    def run_cmd(cmd, env=None):
        e = os.environ.copy()
        if env:
            e.update(env)
        start = time.perf_counter_ns()
        try:
            res = subprocess.run(cmd, cwd=td, env=e, capture_output=True, text=True, timeout=int(os.environ.get("TIMEOUT", 120)))
            dur_ms = (time.perf_counter_ns() - start) / 1_000_000
            last_line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
            if res.returncode != 0:
                return "crash/error", f"<exit {res.returncode}>"
            return f"{dur_ms:6.1f} ms", last_line
        except subprocess.TimeoutExpired:
            return "timeout", "<timeout>"

    aot_bins = {}
    print("Building Lust AOT binaries (--release-fast)...", file=sys.stderr)
    for prog in programs:
        bin_path = Path(td) / f"{prog}_aot"
        res = subprocess.run([str(lust_bin), "build", str(suite_dir / f"{prog}.lust"), "-o", str(bin_path), "--release-fast"], capture_output=True, text=True)
        if res.returncode == 0:
            aot_bins[prog] = bin_path
            # One warmup execution to prime dyld / code signing cache
            subprocess.run([str(bin_path)], capture_output=True)
        else:
            print(f"Failed to build AOT for {prog}: {res.stderr}", file=sys.stderr)

    print(f"{'program':<10} {'rust':>9} {'lust-aot':>10} {'luajit':>9} {'lust-jit':>10} {'lust-vm':>10} {'lua':>9}   outputs")

    for prog in programs:
        t_rust, o_rust = run_cmd([str(rust_bin), prog])
        if prog in aot_bins:
            t_aot, o_aot = run_cmd([str(aot_bins[prog])])
        else:
            t_aot, o_aot = "n/a", ""
        t_jit, o_jit = run_cmd([str(lust_bin), str(suite_dir / f"{prog}.lust")], {"LUST_JIT": "1"})
        t_vm, o_vm = run_cmd([str(lust_bin), str(suite_dir / f"{prog}.lust")], {"LUST_JIT": "0"})
        t_lj, o_lj = run_cmd(["luajit", str(suite_dir / f"{prog}.lua")])
        t_lua, o_lua = run_cmd(["lua", str(suite_dir / f"{prog}.lua")])
        
        outputs = [("rust", o_rust), ("aot", o_aot), ("jit", o_jit), ("luajit", o_lj), ("vm", o_vm), ("lua", o_lua)]
        reference = outputs[0][1]
        def match_val(a, b):
            if a == b:
                return True
            try:
                fa, fb = float(a), float(b)
                import math
                return math.isclose(fa, fb, rel_tol=1e-6)
            except ValueError:
                return False
        diff = [name for name, output in outputs[1:] if not match_val(reference, output)]
        status = "same" if not diff else "DIFF: rust!=" + ",".join(diff)
        print(f"{prog:<10} {t_rust:>9} {t_aot:>10} {t_lj:>9} {t_jit:>10} {t_vm:>10} {t_lua:>9}   {status}")
EOF

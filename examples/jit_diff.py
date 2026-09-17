#!/usr/bin/env python3
"""Differential check: interpreter vs JIT.

Runs every example program twice through the same `lust` binary, once with
`LUST_JIT=0` (interpreter only, the ground truth) and once with the JIT
enabled, and reports any difference in stdout, stderr, or exit code.

    ./examples/jit_diff.py                 # uses target/release/lust
    ./examples/jit_diff.py --bin path/to/lust [--verbose] [paths...]

Use a release build: debug builds print JIT trace logs to stdout.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIRS = ["examples/basic", "examples/advanced", "examples/traits", "examples/programs"]


def run(binary, path, jit, timeout):
    env = dict(os.environ, LUST_JIT="1" if jit else "0")
    try:
        proc = subprocess.run(
            [binary, str(path)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return "timeout", b"", b""


def collect(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.lust")))
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(ROOT / "target/release/lust"))
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("paths", nargs="*", default=DEFAULT_DIRS)
    args = ap.parse_args()

    if not Path(args.bin).exists():
        sys.exit(f"binary not found: {args.bin} (run `cargo build --release`)")

    files = collect(args.paths)
    mismatches = []
    for path in files:
        interp = run(args.bin, path, False, args.timeout)
        jit = run(args.bin, path, True, args.timeout)
        ok = interp == jit
        status = "ok  " if ok else "DIFF"
        if args.verbose or not ok:
            print(f"[{status}] {path}  interp={interp[0]} jit={jit[0]}")
        if not ok:
            mismatches.append((path, interp, jit))
            if interp[1] != jit[1]:
                print("  --- stdout differs (interp / jit) ---")
                print("  " + interp[1].decode(errors="replace").rstrip().replace("\n", "\n  ")[-2000:])
                print("  ---")
                print("  " + jit[1].decode(errors="replace").rstrip().replace("\n", "\n  ")[-2000:])
            if interp[2] != jit[2]:
                print("  --- stderr differs (interp / jit) ---")
                print("  " + interp[2].decode(errors="replace").rstrip().replace("\n", "\n  ")[-2000:])
                print("  ---")
                print("  " + jit[2].decode(errors="replace").rstrip().replace("\n", "\n  ")[-2000:])

    print(f"\n{len(files) - len(mismatches)}/{len(files)} programs agree between interpreter and JIT")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()

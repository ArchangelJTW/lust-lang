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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIRS = [
    "examples/basic",
    "examples/advanced",
    "examples/traits",
    "examples/programs",
    "examples/jit",
]
# Repository directories the examples read relative to the working
# directory. Each run gets a pristine copy, so a program that rewrites its
# input (examples/programs/profile_cli.lust) cannot change what the next
# engine, or the next run, sees.
SEED_DIRS = ["profiles"]


def run(binary, path, jit, timeout):
    """Run one program under one engine, in a private working directory.

    Programs that write files (examples/programs/profile_cli.lust,
    examples/basic/18_io.lust, 19_os.lust) behave differently on a second
    run, so sharing a directory between the two engines makes the first one
    to run change what the second one sees — and, when the directory is the
    repository, leaves a modified tracked file behind. Module resolution is
    relative to the script's own directory, not the working directory, so a
    scratch one is safe.
    """
    env = dict(os.environ, LUST_JIT="1" if jit else "0")
    script = Path(path).resolve()
    try:
        with tempfile.TemporaryDirectory(prefix="lust-jit-diff-") as workdir:
            for name in SEED_DIRS:
                source = ROOT / name
                if source.is_dir():
                    shutil.copytree(source, Path(workdir) / name)
            proc = subprocess.run(
                [binary, str(script)],
                cwd=workdir,
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

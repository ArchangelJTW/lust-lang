#!/usr/bin/env python3
"""Compare two Lust binaries with alternating runs and exact output checks.

Build and save the baseline first, then build the candidate. For example:
    taskset -c 2 python3 benchmarks/suite/compare.py /tmp/lust-before target/release/lust

Times include process startup and frontend compilation, like run.sh, but use a
monotonic timer without shell/Python timer subprocesses around each invocation.
"""

import argparse
import json
from pathlib import Path
import os
import platform
import statistics
import subprocess
import tempfile
import time


PROGRAMS = (
    "fields", "array", "sieve", "calls", "methods", "strings", "fib", "nested",
    "floatmath", "tree",
)
SUITE = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def run(binary, program, mode, timeout):
    env = os.environ.copy()
    env["LUST_JIT"] = "1" if mode == "jit" else "0"
    with tempfile.TemporaryDirectory(prefix="lust-bench-") as work:
        started = time.perf_counter_ns()
        try:
            result = subprocess.run(
                [str(binary), str(SUITE / f"{program}.lust")],
                cwd=work, env=env, capture_output=True, check=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise SystemExit(f"{binary}: {program}/{mode} timed out after {timeout}s") from error
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"{binary}: {program}/{mode} exited with {error.returncode}\n"
                + error.stdout.decode(errors="replace")
                + error.stderr.decode(errors="replace")
            ) from error
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return elapsed_ms, result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--warmups", type=nonnegative_int, default=1)
    parser.add_argument("--timeout", type=positive_int, default=120, help="seconds per invocation")
    parser.add_argument("--mode", choices=("vm", "jit", "both"), default="both")
    parser.add_argument("--programs", nargs="+", choices=PROGRAMS, default=PROGRAMS)
    parser.add_argument("--json", type=Path, help="save individual samples and medians")
    args = parser.parse_args()
    binaries = {"before": args.before.expanduser().resolve(), "after": args.after.expanduser().resolve()}
    for binary in binaries.values():
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f"not an executable file: {binary}")
    modes = ("vm", "jit") if args.mode == "both" else (args.mode,)
    report = {
        "before": str(binaries["before"]), "after": str(binaries["after"]),
        "platform": platform.platform(), "runs": args.runs, "warmups": args.warmups,
        "results": [],
    }
    print(f"{'program':<12} {'mode':<4} {'before (ms)':>12} {'after (ms)':>12} {'speedup':>9}   outputs", flush=True)
    for program in args.programs:
        expected = None
        for mode in modes:
            samples = {"before": [], "after": []}
            for iteration in range(args.warmups + args.runs):
                # Neither binary always gets the first (or warmer) run.
                order = ("before", "after") if iteration % 2 == 0 else ("after", "before")
                for label in order:
                    elapsed, output = run(binaries[label], program, mode, args.timeout)
                    if expected is None:
                        expected = output
                    elif output != expected:
                        raise SystemExit(
                            f"OUTPUT MISMATCH: {program}/{mode}, {label}\n"
                            f"expected: {expected!r}\nactual:   {output!r}"
                        )
                    if iteration >= args.warmups:
                        samples[label].append(elapsed)
            before = statistics.median(samples["before"])
            after = statistics.median(samples["after"])
            speedup = before / after
            print(f"{program:<12} {mode:<4} {before:>12.3f} {after:>12.3f} {speedup:>8.2f}x   same", flush=True)
            report["results"].append({
                "program": program, "mode": mode, "samples_ms": samples,
                "before_ms": before, "after_ms": after, "speedup": speedup,
            })
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Differential tester for the Lust JIT.

Runs every case in `cases.py` twice -- once with `jit = true` and once with
`jit = false` -- and classifies the outcomes.  The interesting output is not
"did it pass" but *where the two modes disagree*, because that isolates JIT
bugs from front-end bugs.

Verdicts per case:

  MATCH_OK      both modes agree and match the expected value
  JIT_WRONG     interpreter is correct, JIT produced a different value
  JIT_HANG      interpreter finished, JIT did not terminate
  JIT_CRASH     interpreter finished, JIT panicked / aborted
  JIT_ERROR     interpreter finished, JIT raised a runtime error
  INTERP_WRONG  interpreter disagrees with the expected value
  BOTH_WRONG    both modes agree with each other but not with expected
  FRONTEND      both modes fail identically (parse / type / compile error)
  JIT_ONLY_OK   the JIT is right and the interpreter is wrong (unlikely)

Usage:
    tools/jit-diff/jitdiff.py
    tools/jit-diff/jitdiff.py --filter shape/for/d2 --timeout 3
    tools/jit-diff/jitdiff.py --json out.json --markdown out.md
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import Case, all_cases  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_LUST = REPO / "target" / "release" / "lust"

# Outcome kinds for a single run.
OK = "ok"
WRONG = "wrong"
HANG = "hang"
CRASH = "crash"
ERROR = "error"
EMPTY = "empty"

ANSI = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class Run:
    kind: str
    value: int | None
    exit_code: int | None
    detail: str

    def summary(self) -> str:
        if self.kind == OK:
            return f"ok({self.value})"
        if self.kind == WRONG:
            return f"wrong({self.value})"
        if self.kind == EMPTY:
            return "no-output"
        return self.kind


@dataclass
class Result:
    name: str
    tags: list[str]
    expected: int
    verdict: str
    jit_on: Run
    jit_off: Run
    source: str


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def classify(proc_out: str, proc_err: str, code: int | None, expected: int, timed_out: bool) -> Run:
    combined = strip_ansi(proc_out + proc_err)

    if timed_out:
        return Run(HANG, None, None, "no termination within timeout")

    if "panicked at" in combined or (code is not None and code < 0) or code in (134, 139):
        first = next(
            (ln.strip() for ln in combined.splitlines() if "panicked at" in ln or "overflow" in ln),
            "abnormal termination",
        )
        return Run(CRASH, None, code, first)

    lines = [ln.strip() for ln in strip_ansi(proc_out).splitlines() if ln.strip()]

    if code != 0 or not lines:
        err_lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
        detail = ""
        for ln in err_lines:
            if "error" in ln.lower():
                detail = ln
                break
        if not detail:
            detail = err_lines[0] if err_lines else "no output"
        if not lines:
            return Run(ERROR if code != 0 else EMPTY, None, code, detail[:200])
        return Run(ERROR, None, code, detail[:200])

    last = lines[-1]
    try:
        value = int(last)
    except ValueError:
        return Run(ERROR, None, code, f"non-integer output: {last[:120]}")

    return Run(OK if value == expected else WRONG, value, code, "")


def run_case(lust: Path, workdir: Path, case: Case, timeout: float) -> Run:
    script = workdir / "case.lust"
    script.write_text(case.source)
    try:
        proc = subprocess.run(
            [str(lust), str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workdir),
        )
    except subprocess.TimeoutExpired:
        return classify("", "", None, case.expected, True)
    return classify(proc.stdout, proc.stderr, proc.returncode, case.expected, False)


def verdict_for(on: Run, off: Run) -> str:
    off_good = off.kind == OK
    on_good = on.kind == OK

    if on_good and off_good:
        return "MATCH_OK"

    if off_good and not on_good:
        return {
            WRONG: "JIT_WRONG",
            HANG: "JIT_HANG",
            CRASH: "JIT_CRASH",
            ERROR: "JIT_ERROR",
            EMPTY: "JIT_ERROR",
        }[on.kind]

    if on_good and not off_good:
        return "JIT_ONLY_OK"

    # Neither mode produced the expected value.
    if on.kind == WRONG and off.kind == WRONG:
        return "BOTH_WRONG" if on.value == off.value else "JIT_WRONG"
    if on.kind == off.kind and on.kind in (ERROR, EMPTY):
        return "FRONTEND"
    if off.kind in (ERROR, EMPTY):
        return "INTERP_WRONG" if on.kind != off.kind else "FRONTEND"
    return "INTERP_WRONG"


def worker(args) -> Result:
    lust, case, timeout, root, index = args
    on_dir = root / f"on_{index}"
    off_dir = root / f"off_{index}"
    for d, jit in ((on_dir, "true"), (off_dir, "false")):
        d.mkdir(parents=True, exist_ok=True)
        (d / "lust-config.toml").write_text(
            f'[settings]\nstdlib_modules = []\njit = {jit}\n'
        )

    # Run the interpreter first: it is the reference, and it never hangs, so a
    # failure there tells us not to trust anything else about the case.
    off = run_case(lust, off_dir, case, timeout)
    on = run_case(lust, on_dir, case, timeout)

    shutil.rmtree(on_dir, ignore_errors=True)
    shutil.rmtree(off_dir, ignore_errors=True)

    return Result(
        name=case.name,
        tags=list(case.tags),
        expected=case.expected,
        verdict=verdict_for(on, off),
        jit_on=on,
        jit_off=off,
        source=case.source,
    )


SEVERITY = [
    "JIT_CRASH",
    "JIT_HANG",
    "JIT_WRONG",
    "JIT_ERROR",
    "INTERP_WRONG",
    "BOTH_WRONG",
    "FRONTEND",
    "JIT_ONLY_OK",
    "MATCH_OK",
]


def render_markdown(results: list[Result], lust: Path, timeout: float) -> str:
    counts = Counter(r.verdict for r in results)
    out: list[str] = []
    out.append("# Lust JIT differential report\n")
    out.append(f"- binary: `{lust}`")
    out.append(f"- cases: {len(results)}")
    out.append(f"- per-run timeout: {timeout}s\n")

    out.append("## Verdict summary\n")
    out.append("| verdict | count | meaning |")
    out.append("|---|---:|---|")
    meanings = {
        "MATCH_OK": "both modes correct",
        "JIT_WRONG": "**JIT silently produced a different value**",
        "JIT_HANG": "**JIT did not terminate**",
        "JIT_CRASH": "**JIT panicked or aborted the process**",
        "JIT_ERROR": "JIT raised a runtime error the interpreter did not",
        "INTERP_WRONG": "interpreter disagrees with expected value",
        "BOTH_WRONG": "both modes agree, both differ from expected",
        "FRONTEND": "parse/type/compile error in both modes",
        "JIT_ONLY_OK": "JIT correct, interpreter wrong",
    }
    for verdict in SEVERITY:
        if counts.get(verdict):
            out.append(f"| {verdict} | {counts[verdict]} | {meanings[verdict]} |")
    out.append("")

    # Which tags are most affected.
    tag_fail: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        for t in r.tags:
            tag_fail[t][r.verdict] += 1
    interesting = [
        (t, c) for t, c in tag_fail.items() if sum(v for k, v in c.items() if k != "MATCH_OK")
    ]
    if interesting:
        out.append("## Failures by tag\n")
        out.append("| tag | total | ok | jit_wrong | jit_hang | jit_crash | other |")
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for t, c in sorted(interesting, key=lambda kv: -sum(kv[1].values())):
            total = sum(c.values())
            other = total - c["MATCH_OK"] - c["JIT_WRONG"] - c["JIT_HANG"] - c["JIT_CRASH"]
            out.append(
                f"| `{t}` | {total} | {c['MATCH_OK']} | {c['JIT_WRONG']} | "
                f"{c['JIT_HANG']} | {c['JIT_CRASH']} | {other} |"
            )
        out.append("")

    # Smallest failing case per verdict is the most useful thing for debugging.
    out.append("## Smallest failing case per verdict\n")
    for verdict in SEVERITY:
        if verdict == "MATCH_OK":
            continue
        group = [r for r in results if r.verdict == verdict]
        if not group:
            continue
        smallest = min(group, key=lambda r: (len(r.source), r.name))
        out.append(f"### {verdict} -- `{smallest.name}`\n")
        out.append(f"expected `{smallest.expected}`, "
                   f"jit=off `{smallest.jit_off.summary()}`, "
                   f"jit=on `{smallest.jit_on.summary()}`\n")
        if smallest.jit_on.detail:
            out.append(f"> {smallest.jit_on.detail}\n")
        out.append("```lust")
        out.append(smallest.source.rstrip())
        out.append("```\n")

    failures = [r for r in results if r.verdict != "MATCH_OK"]
    if failures:
        out.append("## All failing cases\n")
        out.append("| case | verdict | expected | jit=off | jit=on |")
        out.append("|---|---|---:|---|---|")
        order = {v: i for i, v in enumerate(SEVERITY)}
        for r in sorted(failures, key=lambda r: (order[r.verdict], r.name)):
            out.append(
                f"| `{r.name}` | {r.verdict} | {r.expected} | "
                f"{r.jit_off.summary()} | {r.jit_on.summary()} |"
            )
        out.append("")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Lust JIT differential tester")
    ap.add_argument("--lust", type=Path, default=DEFAULT_LUST, help="path to the lust binary")
    ap.add_argument("--filter", default="", help="only run cases whose name contains this substring")
    ap.add_argument("--tag", default="", help="only run cases carrying this tag")
    ap.add_argument("--timeout", type=float, default=5.0, help="per-run timeout in seconds")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--json", type=Path, default=HERE / "report.json")
    ap.add_argument("--markdown", type=Path, default=HERE / "report.md")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args()

    if not args.lust.exists():
        print(f"error: lust binary not found at {args.lust}", file=sys.stderr)
        print("build it first:  cargo build --release", file=sys.stderr)
        return 2

    cases = [c for c in all_cases() if args.filter in c.name]
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
    if not cases:
        print("no cases matched", file=sys.stderr)
        return 2

    print(f"running {len(cases)} cases x 2 modes, {args.jobs} jobs, {args.timeout}s timeout")

    results: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="jitdiff-") as tmp:
        root = Path(tmp)
        payload = [(args.lust, c, args.timeout, root, i) for i, c in enumerate(cases)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for n, res in enumerate(pool.map(worker, payload), 1):
                results.append(res)
                if not args.quiet and res.verdict != "MATCH_OK":
                    print(
                        f"  [{res.verdict:<12}] {res.name}  "
                        f"expected={res.expected} off={res.jit_off.summary()} "
                        f"on={res.jit_on.summary()}"
                    )
                if n % 50 == 0:
                    print(f"  ... {n}/{len(cases)}", file=sys.stderr)

    results.sort(key=lambda r: r.name)
    counts = Counter(r.verdict for r in results)

    args.json.write_text(json.dumps([asdict(r) for r in results], indent=2))
    args.markdown.write_text(render_markdown(results, args.lust, args.timeout))

    print("\n=== summary ===")
    for verdict in SEVERITY:
        if counts.get(verdict):
            print(f"  {verdict:<13} {counts[verdict]}")
    print(f"\nwrote {args.markdown} and {args.json}")

    jit_bugs = sum(counts.get(v, 0) for v in ("JIT_WRONG", "JIT_HANG", "JIT_CRASH", "JIT_ERROR"))
    return 1 if jit_bugs else 0


if __name__ == "__main__":
    sys.exit(main())

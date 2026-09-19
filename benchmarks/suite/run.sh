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
TIMEOUT=${TIMEOUT:-120}
ms() { python3 -c 'import time;print(int(time.time()*1000))'; }
has() { command -v "$1" > /dev/null 2>&1; }
# Run one engine over one program; sets T (ms, or "timeout") and O (last line).
timed() {
  local start
  start=$(ms)
  O=$(cd "$WORK" && timeout "$TIMEOUT" "$@" 2>&1 | tail -1)
  local status=${PIPESTATUS[0]}
  T=$(( $(ms) - start ))
  if [ "$status" -eq 124 ]; then
    T="timeout"
    O="<timeout after ${TIMEOUT}s>"
  fi
}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
printf "%-10s %10s %10s %10s %10s   %s\n" program lust-vm lust-jit luajit lua outputs
for f in fields array calls methods strings fib nested floatmath tree; do
  timed env LUST_JIT=0 "$LUST" "$D/$f.lust"; t1=$T; o1=$O
  timed env LUST_JIT=1 "$LUST" "$D/$f.lust"; t2=$T; o2=$O
  t3=-; t4=-; o3=$o2
  if has luajit; then timed luajit "$D/$f.lua"; t3=$T; o3=$O; fi
  if has lua; then timed lua "$D/$f.lua"; t4=$T; fi
  same="same"; [ "$o1" = "$o2" ] || same="LUST DIFFERS"; [ "$o2" = "$o3" ] || same="$same (lua prints $o3)"
  printf "%-10s %8s ms %8s ms %8s ms %8s ms   %s\n" "$f" "$t1" "$t2" "$t3" "$t4" "$same"
done

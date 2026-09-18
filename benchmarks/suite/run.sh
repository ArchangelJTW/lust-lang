#!/bin/bash
# Cross-language suite: Lust interpreter, Lust JIT, LuaJIT, Lua.
# Usage: benchmarks/suite/run.sh [path/to/lust]   (default: target/release/lust)
D=$(cd "$(dirname "$0")" && pwd)
LUST=${1:-"$D/../../target/release/lust"}
ms() { python3 -c 'import time;print(int(time.time()*1000))'; }
has() { command -v "$1" > /dev/null 2>&1; }
printf "%-10s %10s %10s %10s %10s   %s\n" program lust-vm lust-jit luajit lua outputs
for f in fields array calls methods strings fib nested floatmath; do
  s=$(ms); o1=$(LUST_JIT=0 "$LUST" "$D/$f.lust" 2>&1 | tail -1); t1=$(( $(ms) - s ))
  s=$(ms); o2=$("$LUST" "$D/$f.lust" 2>&1 | tail -1); t2=$(( $(ms) - s ))
  t3=-; t4=-; o3=$o2
  if has luajit; then s=$(ms); o3=$(luajit "$D/$f.lua" 2>&1 | tail -1); t3=$(( $(ms) - s )); fi
  if has lua; then s=$(ms); lua "$D/$f.lua" > /dev/null 2>&1; t4=$(( $(ms) - s )); fi
  same="same"; [ "$o1" = "$o2" ] || same="LUST DIFFERS"; [ "$o2" = "$o3" ] || same="$same (lua prints $o3)"
  printf "%-10s %8s ms %8s ms %8s ms %8s ms   %s\n" "$f" "$t1" "$t2" "$t3" "$t4" "$same"
done

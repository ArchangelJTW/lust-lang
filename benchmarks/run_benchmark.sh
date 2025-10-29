#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <benchmark_name>"
  exit 1
fi

benchmark_name="$1"

echo "==LuaJIT=="
time luajit ./benchmarks/"$benchmark_name"/benchmark.lua
echo ""

echo "==Lust=="
time ./target/release/lust ./benchmarks/"$benchmark_name"/benchmark.lust
echo ""

echo "==Python=="
time python3 ./benchmarks/"$benchmark_name"/benchmark.py
echo ""

echo "==Rhai=="
time rhai-run ./benchmarks/"$benchmark_name"/benchmark.rhai

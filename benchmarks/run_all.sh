#!/bin/bash
set -eu
D=$(cd "$(dirname "$0")" && pwd)

echo "================================================================="
echo "        LUST LANG UNIFIED BENCHMARK & PERFORMANCE SUITE          "
echo "================================================================="
echo

echo ">>> [1/2] Running Cross-Language Microbenchmark Suite (10 Programs)..."
echo "-----------------------------------------------------------------"
"$D/suite/run.sh"
echo

echo ">>> [2/2] Running Real-World ACL Computation Benchmark (Oracle Matrix)..."
echo "-----------------------------------------------------------------"
"$D/acl/run.sh"
echo

echo "================================================================="
echo "                  ALL BENCHMARKS COMPLETED                       "
echo "================================================================="

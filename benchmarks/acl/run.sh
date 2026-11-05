#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
python3 benchmarks/acl/oracle.py
rustc --edition 2021 --crate-type cdylib -C opt-level=3 -C panic=abort -C link-arg=-lSystem benchmarks/acl/rules/policy_c.rs -o benchmarks/acl/rules/policy.dylib
rustc --edition 2021 --target wasm32-wasip1 --crate-type cdylib -C opt-level=3 -C panic=abort benchmarks/acl/rules/policy_c.rs -o benchmarks/acl/rules/policy.wasm
cargo build --release -p bench-host --bin acl --bin acl_comparators
mkdir -p results/acl/v2
if cargo build --release --manifest-path crates/bench-host-fork/Cargo.toml --bin acl-fork > results/acl/v2/fork-build.log 2>&1; then
  fork_status=ran
else
  fork_status=build_failed
fi
printf '{"fork":"%s"}\n' "$fork_status" > results/acl/v2/status.json
for n in 1 2 3; do
  ACL_OUTPUT="results/acl/v2/upstream_${n}.json" target/release/acl
  ACL_OUTPUT="results/acl/v2/comparators_${n}.json" target/release/acl_comparators
  if [ "$fork_status" = ran ]; then
    ACL_OUTPUT="results/acl/v2/fork_${n}.json" crates/bench-host-fork/target/release/acl-fork
  fi
done
python3 benchmarks/acl/summarize.py

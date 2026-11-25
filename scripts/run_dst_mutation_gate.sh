#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/run_dst_mutation_gate.sh
# Purpose: Execute 100% Mutation Acceptance Gate (Phase 5, Section 3 of DST Plan)
# Chain: Baseline GREEN -> Inject Mutant -> Confirm Witness > 0 -> Oracle RED
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "=============================================================================="
echo "           LUST DST PHASE 5: MUTATION ACCEPTANCE GATE RUNNER                  "
echo "=============================================================================="
echo "Target repo: ${REPO_ROOT}"
echo "Date: $(date -u)"
echo ""

# 1. Run Gate 5 Invariants Teeth Test Suite
echo ">>> [1/3] Running Formal Invariants with Teeth (INV-FIBER-01..03, INV-RX-01, INV-BUDGET-01)..."
cargo test --test dst_invariants_teeth -- --nocapture
echo "✓ Formal Invariants Teeth passed."
echo ""

# 2. Run Gate 5 Fault Injection Test Suite
echo ">>> [2/3] Running Fault Injection Engine Tests (FaultConfiguration, FaultInjectedVfs)..."
cargo test --test dst_fault_injection -- --nocapture
echo "✓ Fault Injection tests passed."
echo ""

# 3. Run Gate 5 100% Mutation Acceptance Gate
echo ">>> [3/3] Running 100% Mutation Acceptance Gate (M1, M2, M3)..."
cargo test --test dst_mutation_gate -- --nocapture
echo "✓ Mutation Acceptance Gate passed."
echo ""

echo "=============================================================================="
echo "                     GATE 5 VERIFICATION SUMMARY                              "
echo "=============================================================================="
echo "  [M1] Gas Accounting (MUTATION_M1_SKIP_GAS):"
echo "       - Execution Witness: CONFIRMED (>0)"
echo "       - Killing Oracle: INV-BUDGET-01 (RED confirmed)"
echo "  [M2] Panic Finalizer (MUTATION_M2_SKIP_PANIC_FINALIZER):"
echo "       - Execution Witness: CONFIRMED (>0)"
echo "       - Killing Oracle: INV-FIBER-01-EXACTLY-ONCE (RED confirmed)"
echo "  [M3] Rx Closed Check (MUTATION_M3_SKIP_CLOSED_CHECK):"
echo "       - Execution Witness: CONFIRMED (>0)"
echo "       - Killing Oracle: INV-RX-01 (RED confirmed)"
echo "=============================================================================="
echo "Status: GATE 5 PASSED (100% Mutation Kills, Teeth Verified, Fault Injection Verified)"
echo "=============================================================================="

#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/run_dst_tost_equivalence.sh
# Purpose: Execute Two One-Sided T-tests (TOST) Statistical Equivalence Verification
#          across Criterion/Micro-benchmarks B01-B15 (Gate 6).
# Invariant: 95% Confidence Interval for relative difference [Δ_lower, Δ_upper]
#            must lie strictly within [-1.5%, +1.5%].
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "=============================================================================="
echo "          LUST DST PHASE 6: TOST STATISTICAL EQUIVALENCE RUNNER               "
echo "=============================================================================="
echo "  Target Repo:    ${REPO_ROOT}"
echo "  Toolchain:      $(rustc --version)"
echo "  Build Profile:  --release (fat LTO, opt-level 3)"
echo "  Benchmarks:     B01 - B15"
echo "  Acceptance:     95% CI ⊆ [-1.5%, +1.5%]"
echo "  Date:           $(date -u)"
echo "=============================================================================="
echo ""

echo ">>> Building benchmarks in release profile..."
cargo build --release --bin tost_equivalence

echo ""
echo ">>> Executing TOST statistical equivalence measurement suite..."
cargo run --release --bin tost_equivalence

TOST_EXIT_CODE=$?

echo ""
if [[ $TOST_EXIT_CODE -eq 0 ]]; then
    echo "=============================================================================="
    echo "       GATE 6 STATUS: ZERO-COST PERFORMANCE EQUIVALENCE VERIFIED (PASS)       "
    echo "=============================================================================="
    echo "  Two One-Sided T-tests confirmed zero overhead on production hot-paths."
    echo "  All B01-B15 benchmarks passed with 95% CI inside [-1.5%, +1.5%]."
    echo "=============================================================================="
    exit 0
else
    echo "=============================================================================="
    echo "       GATE 6 STATUS: PERFORMANCE EQUIVALENCE FAILED / INCONCLUSIVE           "
    echo "=============================================================================="
    exit "$TOST_EXIT_CODE"
fi

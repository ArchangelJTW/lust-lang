#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/run_dst_campaign.sh
# Purpose: Execute Scaled Deterministic Simulation Testing (DST) Campaign (Gate 6)
# Scope: Multi-worker parallel seed range exploration, fail-stop invariant monitoring,
#        automatic Replay Bundle extraction (.tar) upon failure, and statistical throughput metrics.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Default parameters
DEFAULT_SEEDS=500
SEED_START=1
SEED_END=""
SEED_COUNT=""
WORKERS=""
MODE="smoke"
ARTIFACT_DIR="artifacts/replays"
FAIL_SEED=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            SEEDS_ARG="$2"
            shift 2
            if [[ "$SEEDS_ARG" =~ ^([0-9]+)\.\.([0-9]+)$ ]]; then
                SEED_START="${BASH_REMATCH[1]}"
                SEED_END="${BASH_REMATCH[2]}"
                SEED_COUNT=$(( SEED_END - SEED_START + 1 ))
            elif [[ "$SEEDS_ARG" =~ ^[0-9]+$ ]]; then
                SEED_START=1
                SEED_COUNT="$SEEDS_ARG"
                SEED_END=$(( SEED_START + SEED_COUNT - 1 ))
            else
                echo "Error: Invalid format for --seeds '$SEEDS_ARG'. Expected integer or range (e.g. 500 or 1..5000)."
                exit 1
            fi
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --smoke)
            MODE="smoke"
            shift
            ;;
        --nightly)
            MODE="nightly"
            shift
            ;;
        --artifact-dir)
            ARTIFACT_DIR="$2"
            shift 2
            ;;
        --fail-seed)
            FAIL_SEED="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --seeds <COUNT|START..END>   Number of seeds or range (default: 500)"
            echo "  --workers <N>               Number of parallel worker threads (default: CPU cores)"
            echo "  --mode <smoke|nightly>      Execution mode (default: smoke)"
            echo "  --artifact-dir <DIR>        Artifact directory for failure bundles (default: artifacts/replays)"
            echo "  --fail-seed <SEED>          Inject intentional invariant failure on seed (for testing bundle export)"
            echo "  --help, -h                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use $0 --help for usage information."
            exit 1
            ;;
    esac
done

# Resolve defaults based on mode if not specified
if [[ -z "$SEED_COUNT" ]]; then
    if [[ "$MODE" == "nightly" ]]; then
        SEED_START=1
        SEED_COUNT=100000
        SEED_END=100000
    else
        SEED_START=1
        SEED_COUNT="$DEFAULT_SEEDS"
        SEED_END=$(( SEED_START + SEED_COUNT - 1 ))
    fi
fi

if [[ -z "$WORKERS" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        WORKERS="$(nproc)"
    elif command -v sysctl >/dev/null 2>&1; then
        WORKERS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
    else
        WORKERS=4
    fi
fi

mkdir -p "${ARTIFACT_DIR}"

echo "=============================================================================="
echo "          LUST DST PHASE 6: SCALED CI SIMULATION CAMPAIGN RUNNER              "
echo "=============================================================================="
echo "  Target Repo:    ${REPO_ROOT}"
echo "  Execution Mode: ${MODE}"
echo "  Seed Range:     ${SEED_START} .. ${SEED_END} (${SEED_COUNT} total seeds)"
echo "  Worker Threads: ${WORKERS}"
echo "  Artifacts:      ${ARTIFACT_DIR}"
echo "  Date:           $(date -u)"
echo "=============================================================================="
echo ""

# Export environment variables for the campaign test runner
export DST_SEED_START="${SEED_START}"
export DST_SEED_END="${SEED_END}"
export DST_SEED_COUNT="${SEED_COUNT}"
export DST_WORKERS="${WORKERS}"
export DST_ARTIFACT_DIR="${ARTIFACT_DIR}"
if [[ -n "$FAIL_SEED" ]]; then
    export DST_FAIL_SEED="${FAIL_SEED}"
fi

# Run the compiled test campaign runner
echo ">>> Executing parallel simulation workers across seed range..."
cargo test --test dst_campaign_runner -- --nocapture

CAMPAIGN_EXIT_CODE=$?

echo ""
echo "=============================================================================="
if [[ $CAMPAIGN_EXIT_CODE -eq 0 ]]; then
    echo "                 GATE 6 CAMPAIGN STATUS: GREEN (PASSED)                       "
    echo "=============================================================================="
    echo "  All ${SEED_COUNT} seeds passed with 100% invariant compliance."
    echo "  Zero state leaks, zero budget violations, zero scheduler deadlocks."
    echo "=============================================================================="
    exit 0
else
    echo "                 GATE 6 CAMPAIGN STATUS: RED (FAILED)                         "
    echo "=============================================================================="
    echo "  Invariant failure detected! Autonomous Replay Bundle saved to:"
    echo "    ${ARTIFACT_DIR}/"
    echo "  To reproduce bit-for-bit:"
    echo "    cargo test --test dst_replay_bundle -- --nocapture"
    echo "=============================================================================="
    exit "$CAMPAIGN_EXIT_CODE"
fi

#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/build_pgo.sh
# Task:   NG-W13-3 (Wave 13 / Sub-Echelon 13-E5)
# Purpose: Profile-Guided Optimization (PGO) build pipeline for:
#          1) `lust` CLI / Bytecode VM / Cranelift JIT compiler binary
#          2) `liblust_aot_rt.rlib` Rust AOT runtime library
#
# Supported PGO Modes:
#   --mode=instrumented
#       Two-phase instrumented PGO (default on macOS arm64 / Apple Silicon):
#         Phase 1: Build with `-Cprofile-generate=<pgo-dir>/profraw`
#         Phase 2: Run training workload (`benchmarks/suite/*.lust` under
#                  `--jit=off` and `--jit=cranelift`, plus `lust-aot-rt` driver)
#         Phase 3: Merge raw profiles via `llvm-profdata merge`
#         Phase 4: Build optimized release artifacts with `-Cprofile-use=<merged.profdata>`
#
#   --mode=sample <profile.prof>
#       Sample-Based PGO / AutoFDO (`-Cprofile-sample-use=<profile.prof>`,
#       stabilized in rustc 1.100.0; standard Linux production CI pipeline):
#         # Example Linux perf + llvm-profgen capture workflow:
#         #   perf record -b -e cycles:u -o lust.perf.data -- \
#         #     ./target/release/lust run benchmarks/suite/sieve.lust --jit=off
#         #   llvm-profgen --binary=./target/release/lust \
#         #     --perfdata=lust.perf.data --output=lust.AutoFDO.prof
#         #   ./scripts/build_pgo.sh --mode=sample lust.AutoFDO.prof
#
# Invariants:
#   - Offline & locked builds only (`cargo build --release --offline --locked`)
#   - Never overwrites or modifies `results/bench_suite_perf_t{1,2,3,4}/` or
#     `results/eval_perf_t{1,2,3,4}/`
#   - Portable Bash 3.2+ (compatible with macOS `/bin/bash` and Linux)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE=""
PROFILE_PATH=""
COMPONENT="all"
TARGET_DIR="${REPO_ROOT}/target/pgo"
PGO_DIR=""
SUITE_DIR="${REPO_ROOT}/benchmarks/suite"
OUT_DIR=""
TRAINING_SCRIPTS_CSV=""
DRY_RUN=0
CHECK_POWER=0

ALL_SUITE_SCRIPTS=(
    "array"
    "calls"
    "fib"
    "fields"
    "floatmath"
    "methods"
    "nested"
    "sieve"
    "strings"
    "tree"
)

usage() {
    cat <<'EOF'
Usage: scripts/build_pgo.sh [options]

Modes:
  --mode=instrumented                Two-phase PGO (-Cprofile-generate -> train ->
                                     llvm-profdata merge -> -Cprofile-use).
                                     Recommended on macOS arm64 (aarch64-apple-darwin).
  --mode=sample <profile.prof>       Sample-Based PGO / AutoFDO (-Cprofile-sample-use)
                                     from rustc 1.100.0. Recommended on Linux with
                                     perf LBR + llvm-profgen.
  --mode=auto                        Selects instrumented on macOS (Darwin), or sample
                                     when --profile is supplied on Linux.

Options:
  --profile <path>                   Explicit path to .prof sample profile (sample mode)
  --component <all|lust|aot-rt>      Artifacts to build (default: all)
  --target-dir <dir>                 Isolated Cargo target directory (default: target/pgo)
  --pgo-dir <dir>                    Directory for profraw/profdata artifacts
                                     (default: <target-dir>/pgo-data)
  --suite-dir <dir>                  Directory containing training *.lust benchmarks
                                     (default: benchmarks/suite)
  --training-scripts <csv>           Comma-separated subset of suite names without .lust
                                     (default: all 10 suite benchmarks)
  --out-dir <dir>                    Optional output directory for final artifacts & manifest
  --check-power                      Verify AC power / lowpowermode 0 on macOS before build
  --dry-run                          Validate environment & print exact commands without building
  -h, --help                         Show this help message

Linux Sample-PGO (AutoFDO) capture example:
  perf record -b -e cycles:u -o lust.perf.data -- \
    ./target/release/lust run benchmarks/suite/sieve.lust --jit=off
  llvm-profgen --binary=./target/release/lust \
    --perfdata=lust.perf.data --output=lust.AutoFDO.prof
  ./scripts/build_pgo.sh --mode=sample lust.AutoFDO.prof

macOS arm64 Instrumented PGO example:
  ./scripts/build_pgo.sh --mode=instrumented --check-power
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

guard_protected_results_path() {
    local label="$1"
    local path_val="$2"
    if [[ -z "${path_val}" ]]; then
        return 0
    fi
    case "${path_val}" in
        *results/bench_suite_perf_t[1-4]*|*results/eval_perf_t[1-4]*)
            die "${label} ('${path_val}') targets a protected historical benchmark directory (results/*_perf_t{1,2,3,4}/). Refusing to overwrite."
            ;;
    esac
}

resolve_abs_path() {
    local input_path="$1"
    if [[ -d "${input_path}" ]]; then
        (cd "${input_path}" && pwd)
        return 0
    fi
    local dir_part
    local base_part
    dir_part="$(dirname "${input_path}")"
    base_part="$(basename "${input_path}")"
    if [[ -d "${dir_part}" ]]; then
        printf '%s/%s\n' "$(cd "${dir_part}" && pwd)" "${base_part}"
        return 0
    fi
    case "${input_path}" in
        /*) printf '%s\n' "${input_path}" ;;
        *)  printf '%s/%s\n' "$(pwd)" "${input_path}" ;;
    esac
}

find_llvm_profdata() {
    local host_triple="$1"
    if [[ -n "${LLVM_PROFDATA:-}" ]]; then
        if [[ -x "${LLVM_PROFDATA}" ]]; then
            printf '%s\n' "${LLVM_PROFDATA}"
            return 0
        fi
        die "LLVM_PROFDATA='${LLVM_PROFDATA}' is not executable"
    fi

    local sysroot
    sysroot="$(rustc --print sysroot)"
    local sysroot_profdata="${sysroot}/lib/rustlib/${host_triple}/bin/llvm-profdata"
    if [[ -x "${sysroot_profdata}" ]]; then
        printf '%s\n' "${sysroot_profdata}"
        return 0
    fi

    if command -v llvm-profdata >/dev/null 2>&1; then
        command -v llvm-profdata
        return 0
    fi

    if command -v xcrun >/dev/null 2>&1; then
        local xcrun_profdata
        if xcrun_profdata="$(xcrun --find llvm-profdata 2>/dev/null)" && [[ -x "${xcrun_profdata}" ]]; then
            printf '%s\n' "${xcrun_profdata}"
            return 0
        fi
    fi

    return 1
}

check_macos_power_mode() {
    local os_name="$1"
    if [[ "${os_name}" != "Darwin" ]]; then
        echo "[power] Non-macOS host (${os_name}); skipping pmset check."
        return 0
    fi
    if ! command -v pmset >/dev/null 2>&1; then
        die "pmset not found on Darwin while --check-power was requested"
    fi
    if ! pmset -g | grep -Eq 'lowpowermode[[:space:]]+0'; then
        die "macOS Low Power Mode is enabled (expected 'lowpowermode 0' in 'pmset -g'). Disable Low Power Mode before benchmarking."
    fi
    echo "[power] Verified macOS pmset: lowpowermode 0"
}

run_or_echo() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[dry-run] '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

run_with_rustflags() {
    local flags="$1"
    shift
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[dry-run] RUSTFLAGS=%q ' "${flags}"
        printf '%q ' "$@"
        printf '\n'
    else
        RUSTFLAGS="${flags}" "$@"
    fi
}

build_lust_cli() {
    local rustflags="$1"
    echo "[build] Building 'lust' CLI/VM binary (target=${HOST_TRIPLE}, offline, locked)..."
    run_with_rustflags "${rustflags}" \
        cargo build --release --offline --locked \
        --manifest-path "${REPO_ROOT}/Cargo.toml" \
        --target "${HOST_TRIPLE}" \
        --target-dir "${TARGET_DIR}" \
        --bin lust
}

build_aot_rt_rlib() {
    local pgo_flag="$1"
    local output_rlib="$2"
    local rt_lib_rs="${REPO_ROOT}/crates/lust-aot-rt/src/lib.rs"
    if [[ ! -f "${rt_lib_rs}" ]]; then
        die "Missing lust-aot-rt source at ${rt_lib_rs}"
    fi
    echo "[build] Building 'liblust_aot_rt.rlib' (${pgo_flag}) -> ${output_rlib}..."
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        mkdir -p "$(dirname "${output_rlib}")"
    fi
    run_or_echo \
        rustc --edition=2024 \
        --crate-name lust_aot_rt \
        --crate-type=rlib \
        -C opt-level=3 \
        -C panic=abort \
        -C embed-bitcode=yes \
        -C codegen-units=1 \
        --cfg 'feature="std"' \
        -C "${pgo_flag}" \
        "${rt_lib_rs}" \
        -o "${output_rlib}"
}

build_aot_rt_cargo() {
    local rustflags="$1"
    echo "[build] Building workspace crate 'lust-aot-rt' via Cargo (target=${HOST_TRIPLE}, offline, locked)..."
    run_with_rustflags "${rustflags}" \
        cargo build --release --offline --locked \
        --manifest-path "${REPO_ROOT}/Cargo.toml" \
        --target "${HOST_TRIPLE}" \
        --target-dir "${TARGET_DIR}" \
        -p lust-aot-rt
}

train_lust_binary() {
    local lust_bin="$1"
    local profraw_dir="$2"
    shift 2
    local scripts=("$@")

    echo "[train] Running Lust training workload (${#scripts[@]} suite benchmarks x 2 modes: --jit=off & --jit=cranelift)..."
    local script_name
    local script_path
    for script_name in "${scripts[@]}"; do
        script_path="${SUITE_DIR}/${script_name}.lust"
        if [[ ! -f "${script_path}" ]]; then
            die "Training benchmark script not found: ${script_path}"
        fi
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            printf '[dry-run] LLVM_PROFILE_FILE=%q %q run %q --jit=off\n' \
                "${profraw_dir}/lust-vm-${script_name}-%p-%m.profraw" "${lust_bin}" "${script_path}"
            printf '[dry-run] LLVM_PROFILE_FILE=%q %q run %q --jit=cranelift\n' \
                "${profraw_dir}/lust-jit-${script_name}-%p-%m.profraw" "${lust_bin}" "${script_path}"
        else
            echo "  -> ${script_name}.lust (--jit=off)"
            LLVM_PROFILE_FILE="${profraw_dir}/lust-vm-${script_name}-%p-%m.profraw" \
                "${lust_bin}" run "${script_path}" --jit=off >/dev/null
            echo "  -> ${script_name}.lust (--jit=cranelift)"
            LLVM_PROFILE_FILE="${profraw_dir}/lust-jit-${script_name}-%p-%m.profraw" \
                "${lust_bin}" run "${script_path}" --jit=cranelift >/dev/null
        fi
    done
}

train_aot_rt_driver() {
    local gen_rlib="$1"
    local profraw_dir="$2"
    local work_dir="${PGO_DIR}/aot_rt_train"
    local driver_rs="${work_dir}/driver.rs"
    local driver_bin="${work_dir}/driver"

    echo "[train] Running lust-aot-rt instrumented training driver..."
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[dry-run] rustc --edition=2024 -C opt-level=3 -C panic=abort -C profile-generate=%q --extern lust_aot_rt=%q %q -o %q\n' \
            "${profraw_dir}" "${gen_rlib}" "${driver_rs}" "${driver_bin}"
        printf '[dry-run] LLVM_PROFILE_FILE=%q %q\n' \
            "${profraw_dir}/lust-aot-rt-%p-%m.profraw" "${driver_bin}"
        return 0
    fi

    mkdir -p "${work_dir}"
    cat > "${driver_rs}" <<'EOF'
extern crate lust_aot_rt;

use lust_aot_rt::{
    AotCtx, AotMap, AotValue, EffectHandlerArm, LustChunkArena, handle_tail_resumptive_typed,
    perform_effect, resume_effect, scope_register_finalizer, with_aot_scope,
};

fn main() {
    let mut ctx = AotCtx::with_memory_limit(16 * 1024 * 1024);

    let mut acc: i64 = 0;
    let map: AotMap<i64, AotValue> = AotMap::new();
    for i in 0..2000_i64 {
        let _ = ctx.budget_alloc.try_charge(16);
        let key = i % 128;
        let _ = map.set(key, AotValue::Int(i.wrapping_mul(3)), &ctx.budget_alloc);
        if let Ok(AotValue::Int(v)) = map.get(&key) {
            acc = acc.wrapping_add(v);
        }
    }

    let arm = EffectHandlerArm::from_fn("PgoTrainEff", |c, args| {
        let x = match args.first() {
            Some(AotValue::Int(v)) => *v,
            _ => 0,
        };
        resume_effect(c, AotValue::Int(x.wrapping_add(1)))
    });

    let eff_val = handle_tail_resumptive_typed(
        &mut ctx,
        vec![arm],
        |c| {
            let mut local_sum: i64 = 0;
            for j in 0..500_i64 {
                if let Ok(AotValue::Int(r)) = perform_effect(c, "PgoTrainEff", &[AotValue::Int(j)])
                {
                    local_sum = local_sum.wrapping_add(r);
                }
            }
            Ok(local_sum)
        },
        |_abort| Ok(0),
    )
    .unwrap_or(0);

    let scoped_val = with_aot_scope(&mut ctx, |c| {
        let _ = scope_register_finalizer(c, |_c2| Ok(()));
        Ok(eff_val.wrapping_add(acc))
    })
    .unwrap_or(0);

    let arena = LustChunkArena::new(&ctx.budget_alloc);
    let _ = arena.chunk_count();
    println!("{}", scoped_val);
}
EOF

    rustc --edition=2024 \
        -C opt-level=3 \
        -C panic=abort \
        -C "profile-generate=${profraw_dir}" \
        --extern "lust_aot_rt=${gen_rlib}" \
        "${driver_rs}" \
        -o "${driver_bin}"

    LLVM_PROFILE_FILE="${profraw_dir}/lust-aot-rt-%p-%m.profraw" \
        "${driver_bin}" >/dev/null
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode=*)
            MODE="${1#--mode=}"
            shift
            if [[ "${MODE}" == "sample" && $# -gt 0 && "$1" != --* ]]; then
                PROFILE_PATH="$1"
                shift
            fi
            ;;
        --mode)
            [[ $# -ge 2 ]] || die "--mode requires a value (instrumented | sample | auto)"
            MODE="$2"
            shift 2
            if [[ "${MODE}" == "sample" && $# -gt 0 && "$1" != --* ]]; then
                PROFILE_PATH="$1"
                shift
            fi
            ;;
        --profile=*)
            PROFILE_PATH="${1#--profile=}"
            shift
            ;;
        --profile)
            [[ $# -ge 2 ]] || die "--profile requires a path to a .prof sample profile"
            PROFILE_PATH="$2"
            shift 2
            ;;
        --component=*)
            COMPONENT="${1#--component=}"
            shift
            ;;
        --component)
            [[ $# -ge 2 ]] || die "--component requires one of: all, lust, aot-rt"
            COMPONENT="$2"
            shift 2
            ;;
        --target-dir=*)
            TARGET_DIR="${1#--target-dir=}"
            shift
            ;;
        --target-dir)
            [[ $# -ge 2 ]] || die "--target-dir requires a directory path"
            TARGET_DIR="$2"
            shift 2
            ;;
        --pgo-dir=*)
            PGO_DIR="${1#--pgo-dir=}"
            shift
            ;;
        --pgo-dir)
            [[ $# -ge 2 ]] || die "--pgo-dir requires a directory path"
            PGO_DIR="$2"
            shift 2
            ;;
        --suite-dir=*)
            SUITE_DIR="${1#--suite-dir=}"
            shift
            ;;
        --suite-dir)
            [[ $# -ge 2 ]] || die "--suite-dir requires a directory path"
            SUITE_DIR="$2"
            shift 2
            ;;
        --training-scripts=*)
            TRAINING_SCRIPTS_CSV="${1#--training-scripts=}"
            shift
            ;;
        --training-scripts)
            [[ $# -ge 2 ]] || die "--training-scripts requires a comma-separated list"
            TRAINING_SCRIPTS_CSV="$2"
            shift 2
            ;;
        --out-dir=*)
            OUT_DIR="${1#--out-dir=}"
            shift
            ;;
        --out-dir)
            [[ $# -ge 2 ]] || die "--out-dir requires a directory path"
            OUT_DIR="$2"
            shift 2
            ;;
        --check-power)
            CHECK_POWER=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument '$1'. Use --help for usage."
            ;;
    esac
done

OS_NAME="$(uname -s)"
HOST_TRIPLE="$(rustc -vV | awk '/^host:/ {print $2}')"
RUSTC_VERSION="$(rustc --version)"

if [[ -z "${MODE}" || "${MODE}" == "auto" ]]; then
    if [[ "${OS_NAME}" == "Darwin" ]]; then
        MODE="instrumented"
    elif [[ -n "${PROFILE_PATH}" ]]; then
        MODE="sample"
    else
        MODE="instrumented"
    fi
fi

case "${MODE}" in
    instrumented|sample) ;;
    *) die "Unsupported --mode '${MODE}'. Expected 'instrumented', 'sample', or 'auto'." ;;
esac

case "${COMPONENT}" in
    all|lust|aot-rt) ;;
    *) die "Unsupported --component '${COMPONENT}'. Expected 'all', 'lust', or 'aot-rt'." ;;
esac

if [[ -z "${PGO_DIR}" ]]; then
    PGO_DIR="${TARGET_DIR}/pgo-data"
fi

guard_protected_results_path "--target-dir" "${TARGET_DIR}"
guard_protected_results_path "--pgo-dir" "${PGO_DIR}"
guard_protected_results_path "--out-dir" "${OUT_DIR}"

TARGET_DIR="$(resolve_abs_path "${TARGET_DIR}")"
PGO_DIR="$(resolve_abs_path "${PGO_DIR}")"
SUITE_DIR="$(resolve_abs_path "${SUITE_DIR}")"
if [[ -n "${OUT_DIR}" ]]; then
    OUT_DIR="$(resolve_abs_path "${OUT_DIR}")"
fi

if [[ "${CHECK_POWER}" -eq 1 ]]; then
    check_macos_power_mode "${OS_NAME}"
fi

# Verify rustc supports the required PGO codegen flags
RUSTC_C_HELP="$(rustc -C help 2>&1)"
if [[ "${MODE}" == "sample" ]]; then
    printf '%s\n' "${RUSTC_C_HELP}" | grep -q 'profile-sample-use' \
        || die "Active rustc (${RUSTC_VERSION}) does not support -Cprofile-sample-use (requires rustc >= 1.100.0)."
    [[ -n "${PROFILE_PATH}" ]] \
        || die "--mode=sample requires a sample profile path (--mode=sample <profile.prof> or --profile <path>)."
    [[ -f "${PROFILE_PATH}" && -r "${PROFILE_PATH}" ]] \
        || die "Sample profile file does not exist or is not readable: ${PROFILE_PATH}"
    [[ -s "${PROFILE_PATH}" ]] \
        || die "Sample profile file is empty: ${PROFILE_PATH}"
    PROFILE_PATH="$(resolve_abs_path "${PROFILE_PATH}")"
else
    printf '%s\n' "${RUSTC_C_HELP}" | grep -q 'profile-generate' \
        || die "Active rustc (${RUSTC_VERSION}) does not support -Cprofile-generate."
    printf '%s\n' "${RUSTC_C_HELP}" | grep -q 'profile-use' \
        || die "Active rustc (${RUSTC_VERSION}) does not support -Cprofile-use."
    LLVM_PROFDATA_BIN="$(find_llvm_profdata "${HOST_TRIPLE}")" \
        || die "llvm-profdata not found. Install rustup component 'llvm-tools' or set LLVM_PROFDATA."
fi

# Resolve training script list
SELECTED_SCRIPTS=()
if [[ -n "${TRAINING_SCRIPTS_CSV}" ]]; then
    IFS=',' read -r -a SELECTED_SCRIPTS <<< "${TRAINING_SCRIPTS_CSV}"
else
    SELECTED_SCRIPTS=("${ALL_SUITE_SCRIPTS[@]}")
fi

BASE_RUSTFLAGS="${RUSTFLAGS:-}"
FINAL_LUST_BIN="${TARGET_DIR}/${HOST_TRIPLE}/release/lust"
ARTIFACT_DIR="${OUT_DIR:-${PGO_DIR}/artifacts}"
FINAL_AOT_RT_RLIB="${ARTIFACT_DIR}/liblust_aot_rt.rlib"

echo "=============================================================================="
echo "  Lust PGO Build Pipeline (NG-W13-3)"
echo "=============================================================================="
echo "  Host OS / Triple : ${OS_NAME} / ${HOST_TRIPLE}"
echo "  Rustc            : ${RUSTC_VERSION}"
echo "  PGO Mode         : ${MODE}"
echo "  Component        : ${COMPONENT}"
echo "  Target Dir       : ${TARGET_DIR}"
echo "  PGO Work Dir     : ${PGO_DIR}"
echo "  Artifact Dir     : ${ARTIFACT_DIR}"
if [[ "${MODE}" == "sample" ]]; then
    echo "  Sample Profile   : ${PROFILE_PATH}"
else
    echo "  llvm-profdata    : ${LLVM_PROFDATA_BIN}"
    echo "  Suite Dir        : ${SUITE_DIR}"
fi
echo "  Dry Run          : ${DRY_RUN}"
echo "=============================================================================="

if [[ "${MODE}" == "sample" ]]; then
    SAMPLE_FLAG="profile-sample-use=${PROFILE_PATH}"
    SAMPLE_RUSTFLAGS="${BASE_RUSTFLAGS:+${BASE_RUSTFLAGS} }-C${SAMPLE_FLAG}"

    if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "lust" ]]; then
        build_lust_cli "${SAMPLE_RUSTFLAGS}"
    fi
    if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "aot-rt" ]]; then
        build_aot_rt_cargo "${SAMPLE_RUSTFLAGS}"
        build_aot_rt_rlib "${SAMPLE_FLAG}" "${FINAL_AOT_RT_RLIB}"
    fi
else
    PROFRAW_DIR="${PGO_DIR}/profraw"
    MERGED_PROFDATA="${PGO_DIR}/merged.profdata"
    GEN_FLAG="profile-generate=${PROFRAW_DIR}"
    GEN_RUSTFLAGS="${BASE_RUSTFLAGS:+${BASE_RUSTFLAGS} }-C${GEN_FLAG}"
    GEN_AOT_RT_RLIB="${PGO_DIR}/gen/liblust_aot_rt.rlib"

    if [[ "${DRY_RUN}" -eq 0 ]]; then
        rm -rf "${PROFRAW_DIR}" "${MERGED_PROFDATA}"
        mkdir -p "${PROFRAW_DIR}"
    else
        echo "[dry-run] mkdir -p ${PROFRAW_DIR}"
    fi

    echo "--- Phase 1/4: Instrumented Build (-Cprofile-generate) ---"
    if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "lust" ]]; then
        build_lust_cli "${GEN_RUSTFLAGS}"
    fi
    if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "aot-rt" ]]; then
        build_aot_rt_rlib "${GEN_FLAG}" "${GEN_AOT_RT_RLIB}"
    fi

    echo "--- Phase 2/4: Training Workload ---"
    if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "lust" ]]; then
        train_lust_binary "${FINAL_LUST_BIN}" "${PROFRAW_DIR}" "${SELECTED_SCRIPTS[@]}"
    fi
    if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "aot-rt" ]]; then
        train_aot_rt_driver "${GEN_AOT_RT_RLIB}" "${PROFRAW_DIR}"
    fi

    echo "--- Phase 3/4: Merge Profiles (llvm-profdata merge) ---"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[dry-run] %q merge -o %q %q/*.profraw\n' \
            "${LLVM_PROFDATA_BIN}" "${MERGED_PROFDATA}" "${PROFRAW_DIR}"
    else
        PROFRAW_FILES=()
        while IFS= read -r -d '' f; do
            PROFRAW_FILES+=("${f}")
        done < <(find "${PROFRAW_DIR}" -maxdepth 1 -type f -name '*.profraw' -print0 | sort -z)

        if [[ "${#PROFRAW_FILES[@]}" -eq 0 ]]; then
            die "No .profraw files generated in ${PROFRAW_DIR}; training workload did not emit profile data."
        fi
        echo "[merge] Merging ${#PROFRAW_FILES[@]} .profraw file(s) into ${MERGED_PROFDATA}..."
        "${LLVM_PROFDATA_BIN}" merge -o "${MERGED_PROFDATA}" "${PROFRAW_FILES[@]}"
        if [[ ! -s "${MERGED_PROFDATA}" ]]; then
            die "Merged profile file is missing or empty: ${MERGED_PROFDATA}"
        fi
    fi

    echo "--- Phase 4/4: Profile-Optimized Release Build (-Cprofile-use) ---"
    USE_FLAG="profile-use=${MERGED_PROFDATA}"
    USE_RUSTFLAGS="${BASE_RUSTFLAGS:+${BASE_RUSTFLAGS} }-C${USE_FLAG}"
    if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "lust" ]]; then
        build_lust_cli "${USE_RUSTFLAGS}"
    fi
    if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "aot-rt" ]]; then
        build_aot_rt_cargo "${USE_RUSTFLAGS}"
        build_aot_rt_rlib "${USE_FLAG}" "${FINAL_AOT_RT_RLIB}"
    fi
fi

if [[ "${DRY_RUN}" -eq 0 && -n "${OUT_DIR}" ]]; then
    mkdir -p "${OUT_DIR}"
    if [[ ("${COMPONENT}" == "all" || "${COMPONENT}" == "lust") && -x "${FINAL_LUST_BIN}" ]]; then
        cp -f "${FINAL_LUST_BIN}" "${OUT_DIR}/lust"
    fi
    cat > "${OUT_DIR}/pgo-build-manifest.json" <<EOF
{
  "task_id": "NG-W13-3",
  "mode": "${MODE}",
  "component": "${COMPONENT}",
  "host_os": "${OS_NAME}",
  "host_triple": "${HOST_TRIPLE}",
  "rustc_version": "${RUSTC_VERSION}",
  "profile_artifact": "$(if [[ "${MODE}" == "sample" ]]; then printf '%s' "${PROFILE_PATH}"; else printf '%s' "${PGO_DIR}/merged.profdata"; fi)",
  "aot_rt_rlib": "${FINAL_AOT_RT_RLIB}"
}
EOF
fi

echo "=============================================================================="
echo "  PGO Build Pipeline Complete (mode=${MODE}, component=${COMPONENT}, dry_run=${DRY_RUN})"
if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "lust" ]]; then
    echo "  lust binary     : ${FINAL_LUST_BIN}"
fi
if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "aot-rt" ]]; then
    echo "  lust-aot-rt rlib: ${FINAL_AOT_RT_RLIB}"
fi
echo "=============================================================================="

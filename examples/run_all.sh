#!/bin/bash

# Lust Language - Comprehensive Example Runner
# Runs native Lust scripts, Cargo examples, standalone subprojects, and differential checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Defaults
BINARY="${LUST_BIN:-}"
BACKEND="${LUST_BACKEND:-}"
FILTER_JIT=1
CATEGORY_FILTER=""
RUN_CARGO_EXAMPLES=0
RUN_STANDALONE=0
RUN_EVERYTHING=0

# Directories
EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXAMPLES_DIR/.." && pwd)"
BASIC_DIR="$EXAMPLES_DIR/basic"
ADVANCED_DIR="$EXAMPLES_DIR/advanced"
TRAITS_DIR="$EXAMPLES_DIR/traits"
EFFECTS_DIR="$EXAMPLES_DIR/effects"
SCHEMAS_DIR="$EXAMPLES_DIR/schemas"
TESTING_DIR="$EXAMPLES_DIR/testing"
MODULES_DIR="$EXAMPLES_DIR/modules"
PROGRAMS_DIR="$EXAMPLES_DIR/programs"

# Cargo examples in root Cargo.toml (Section A)
CARGO_EXAMPLES=(
	"schema_validation"
	"schema_json"
	"schema_msgpack"
	"rx_observables"
	"rx_operators"
	"rx_backpressure"
	"sim_scheduler"
	"sim_fault_injection"
	"sim_replay_bundle"
	"effects_basic"
	"effects_retry_repeat"
	"effects_acquire_release"
	"effects_di_mocking"
	"effects_supervision"
	"embed_standalone_vm"
)

# Parse command line options
while [[ $# -gt 0 ]]; do
	case "$1" in
		--backend)
			BACKEND="$2"
			shift 2
			;;
		--backend=*)
			BACKEND="${1#*=}"
			shift
			;;
		--binary|--bin|-b)
			BINARY="$2"
			shift 2
			;;
		--binary=*|--bin=*)
			BINARY="${1#*=}"
			shift
			;;
		--category|-c)
			CATEGORY_FILTER="$2"
			shift 2
			;;
		--category=*)
			CATEGORY_FILTER="${1#*=}"
			shift
			;;
		--all)
			CATEGORY_FILTER="ALL"
			shift
			;;
		--cargo|--cargo-examples)
			RUN_CARGO_EXAMPLES=1
			shift
			;;
		--standalone)
			RUN_STANDALONE=1
			shift
			;;
		--everything)
			RUN_EVERYTHING=1
			CATEGORY_FILTER="ALL"
			RUN_CARGO_EXAMPLES=1
			RUN_STANDALONE=1
			shift
			;;
		--no-filter-jit)
			FILTER_JIT=0
			shift
			;;
		--filter-jit)
			FILTER_JIT=1
			shift
			;;
		--help|-h)
			echo "Usage: $0 [options]"
			echo ""
			echo "Options:"
			echo "  --binary, -b <path>     Path to lust executable (default: auto-detected)"
			echo "  --backend <backend>     Backend to pass to 'lust run' (e.g. vm, cranelift, zig)"
			echo "  --category, -c <name>   Run only specified category (BASIC, ADVANCED, TRAITS, EFFECTS,"
			echo "                          SCHEMAS, TESTING, MODULES, PROGRAMS, ALL)"
			echo "  --all                   Run all native .lust categories (including effects, schemas, testing, modules, programs)"
			echo "  --cargo-examples        Run all 15 Cargo examples via 'cargo run --example <name>'"
			echo "  --standalone            Run standalone subprojects (c-ffi, tokio-concurrency, bindgen)"
			echo "  --everything            Run all native scripts, all Cargo examples, and standalone subprojects"
			echo "  --filter-jit            Filter out tracing JIT debug logs (default: enabled)"
			echo "  --no-filter-jit         Keep tracing JIT debug logs"
			echo "  --help, -h              Show this help message"
			exit 0
			;;
		*)
			if [ -x "$1" ] || [ -f "$1" ]; then
				BINARY="$1"
			else
				echo -e "${RED}Unknown argument: $1${NC}"
				echo "Run '$0 --help' for usage."
				exit 1
			fi
			shift
			;;
	esac
done

# Auto-detect binary if not explicitly provided
if [ -z "$BINARY" ]; then
	if [ -x "$ROOT_DIR/target/release/lust" ]; then
		BINARY="$ROOT_DIR/target/release/lust"
	elif [ -x "$ROOT_DIR/target/debug/lust" ]; then
		BINARY="$ROOT_DIR/target/debug/lust"
	elif command -v lust &>/dev/null; then
		BINARY="lust"
	else
		BINARY="$ROOT_DIR/target/debug/lust"
	fi
fi

if [ ! -x "$BINARY" ] && ! command -v "$BINARY" &>/dev/null; then
	echo -e "${RED}Error: lust binary not found at '$BINARY'${NC}"
	echo -e "${YELLOW}Run 'cargo build' or 'cargo build --release' first, or pass --binary <path>${NC}"
	exit 1
fi

# Counters
PASSED=0
FAILED=0
TOTAL=0

# Function to run a single example
run_example() {
	local file=$1
	local category=$2
	local exit_code=0
	local cmd_args=()
	local file_base
	file_base="$(basename "$file")"

	TOTAL=$((TOTAL + 1))

	echo -e "\n${CYAN}[$category]${NC} Running: ${BLUE}$file_base${NC}"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

	if [ -n "$BACKEND" ]; then
		cmd_args=("$BINARY" "run" "$file" "--backend" "$BACKEND")
	else
		cmd_args=("$BINARY" "$file")
	fi

	local output
	# Provide mocked stdin for interactive CLI programs like profile_cli.lust to avoid blocking,
	# and run in a temporary directory so tracked profile.txt is not dirtied
	if [ "$file_base" == "profile_cli.lust" ]; then
		local tmp_workdir
		tmp_workdir="$(mktemp -d -t lust-profile-cli-XXXXXX)"
		if [ -d "$ROOT_DIR/profiles" ]; then
			cp -r "$ROOT_DIR/profiles" "$tmp_workdir/"
		fi
		output=$(printf "TestUser\n25\nCyan\n" | (cd "$tmp_workdir" && "${cmd_args[@]}") 2>&1) || exit_code=$?
		rm -rf "$tmp_workdir"
	else
		output=$("${cmd_args[@]}" 2>&1) || exit_code=$?
	fi

	if [ $exit_code -eq 0 ]; then
		if [ "$FILTER_JIT" -eq 1 ]; then
			# Filter out debug tracing JIT lines
			output=$(printf '%s\n' "$output" | grep -v -E "^\s*(Preamble:|Body:|Postamble:|\[JIT\]|[0-9]{3}: (Guard|Add|Sub|Mul|Div|Move|Jump|Load|Call|Loop))" || true)
		fi
		if [ -n "$output" ]; then
			printf '%s\n' "$output"
		fi
		echo -e "${GREEN}✓ Passed${NC}"
		PASSED=$((PASSED + 1))
		return 0
	else
		# Do not filter output on failure; print full raw diagnostics/backtrace
		if [ -n "$output" ]; then
			printf '%s\n' "$output"
		fi
		echo -e "${RED}✗ Failed (exit code $exit_code)${NC}"
		FAILED=$((FAILED + 1))
		return 1
	fi
}

# Function to run all files in a directory
run_category() {
	local dir=$1
	local category=$2

	if [ ! -d "$dir" ]; then
		echo -e "${YELLOW}Warning: Directory $dir not found, skipping...${NC}"
		return
	fi

	echo -e "\n${YELLOW}═══════════════════════════════════════════${NC}"
	echo -e "${YELLOW}  $category Examples${NC}"
	echo -e "${YELLOW}═══════════════════════════════════════════${NC}"

	# Find all .lust files and sort them
	local files=()
	while IFS= read -r -d '' file; do
		files+=("$file")
	done < <(find "$dir" -maxdepth 1 -name "*.lust" -print0 | sort -z)

	if [ ${#files[@]} -eq 0 ]; then
		echo -e "${YELLOW}No .lust files found in $dir${NC}"
		return
	fi

	for file in "${files[@]}"; do
		run_example "$file" "$category" || true
	done
}

# Function to run a single Cargo example
run_cargo_example() {
	local name=$1
	local exit_code=0
	TOTAL=$((TOTAL + 1))

	echo -e "\n${MAGENTA}[CARGO]${NC} Running example: ${BLUE}$name${NC}"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

	local output
	output=$(cd "$ROOT_DIR" && cargo run --example "$name" 2>&1) || exit_code=$?

	if [ $exit_code -eq 0 ]; then
		if [ -n "$output" ]; then
			printf '%s\n' "$output"
		fi
		echo -e "${GREEN}✓ Passed${NC}"
		PASSED=$((PASSED + 1))
		return 0
	else
		if [ -n "$output" ]; then
			printf '%s\n' "$output"
		fi
		echo -e "${RED}✗ Failed (exit code $exit_code)${NC}"
		FAILED=$((FAILED + 1))
		return 1
	fi
}

# Function to run all Cargo examples
run_all_cargo_examples() {
	echo -e "\n${YELLOW}═══════════════════════════════════════════${NC}"
	echo -e "${YELLOW}  Cargo [[example]] Suite (15 targets)${NC}"
	echo -e "${YELLOW}═══════════════════════════════════════════${NC}"

	for name in "${CARGO_EXAMPLES[@]}"; do
		run_cargo_example "$name" || true
	done
}

# Function to run standalone projects
run_standalone_projects() {
	echo -e "\n${YELLOW}═══════════════════════════════════════════${NC}"
	echo -e "${YELLOW}  Standalone Embedding Subprojects${NC}"
	echo -e "${YELLOW}═══════════════════════════════════════════${NC}"

	# 1. C-FFI
	TOTAL=$((TOTAL + 1))
	echo -e "\n${MAGENTA}[STANDALONE]${NC} Testing: ${BLUE}examples/c-ffi (Makefile test)${NC}"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	local c_ffi_exit=0
	local c_ffi_out
	c_ffi_out=$(make -C "$EXAMPLES_DIR/c-ffi" clean test 2>&1) || c_ffi_exit=$?
	if [ $c_ffi_exit -eq 0 ]; then
		printf '%s\n' "$c_ffi_out"
		echo -e "${GREEN}✓ Passed${NC}"
		PASSED=$((PASSED + 1))
	else
		printf '%s\n' "$c_ffi_out"
		echo -e "${RED}✗ Failed (exit code $c_ffi_exit)${NC}"
		FAILED=$((FAILED + 1))
	fi

	# 2. Tokio Concurrency
	TOTAL=$((TOTAL + 1))
	echo -e "\n${MAGENTA}[STANDALONE]${NC} Testing: ${BLUE}examples/tokio-concurrency${NC}"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	local tokio_exit=0
	local tokio_out
	tokio_out=$(cargo run --manifest-path "$EXAMPLES_DIR/tokio-concurrency/Cargo.toml" 2>&1) || tokio_exit=$?
	if [ $tokio_exit -eq 0 ]; then
		printf '%s\n' "$tokio_out"
		echo -e "${GREEN}✓ Passed${NC}"
		PASSED=$((PASSED + 1))
	else
		printf '%s\n' "$tokio_out"
		echo -e "${RED}✗ Failed (exit code $tokio_exit)${NC}"
		FAILED=$((FAILED + 1))
	fi

	# 3. Bindgen
	TOTAL=$((TOTAL + 1))
	echo -e "\n${MAGENTA}[STANDALONE]${NC} Testing: ${BLUE}examples/bindgen${NC}"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	local bindgen_exit=0
	local bindgen_out
	bindgen_out=$(cargo run --manifest-path "$EXAMPLES_DIR/bindgen/Cargo.toml" 2>&1) || bindgen_exit=$?
	if [ $bindgen_exit -eq 0 ]; then
		printf '%s\n' "$bindgen_out"
		echo -e "${GREEN}✓ Passed${NC}"
		PASSED=$((PASSED + 1))
	else
		printf '%s\n' "$bindgen_out"
		echo -e "${RED}✗ Failed (exit code $bindgen_exit)${NC}"
		FAILED=$((FAILED + 1))
	fi
}

# Main execution banner
echo -e "${CYAN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Lust Language - Example Test Runner     ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════╝${NC}"
echo -e "Binary:  $BINARY"
if [ -n "$BACKEND" ]; then
	echo -e "Backend: $BACKEND"
fi

CATEGORY_UPPER=$(echo "$CATEGORY_FILTER" | tr '[:lower:]' '[:upper:]')
case "$CATEGORY_UPPER" in
	BASIC)
		run_category "$BASIC_DIR" "BASIC"
		;;
	ADVANCED)
		run_category "$ADVANCED_DIR" "ADVANCED"
		;;
	TRAITS)
		run_category "$TRAITS_DIR" "TRAITS"
		;;
	EFFECTS)
		run_category "$EFFECTS_DIR" "EFFECTS"
		;;
	SCHEMAS)
		run_category "$SCHEMAS_DIR" "SCHEMAS"
		;;
	TESTING)
		run_category "$TESTING_DIR" "TESTING"
		;;
	MODULES)
		run_category "$MODULES_DIR" "MODULES"
		;;
	PROGRAMS)
		run_category "$PROGRAMS_DIR" "PROGRAMS"
		;;
	ALL)
		run_category "$BASIC_DIR" "BASIC"
		run_category "$ADVANCED_DIR" "ADVANCED"
		run_category "$TRAITS_DIR" "TRAITS"
		run_category "$EFFECTS_DIR" "EFFECTS"
		run_category "$SCHEMAS_DIR" "SCHEMAS"
		run_category "$TESTING_DIR" "TESTING"
		run_category "$MODULES_DIR" "MODULES"
		run_category "$PROGRAMS_DIR" "PROGRAMS"
		;;
	*)
		if [ $RUN_CARGO_EXAMPLES -eq 0 ] && [ $RUN_STANDALONE -eq 0 ]; then
			# Default: run the standard 35 examples across BASIC, ADVANCED, and TRAITS
			run_category "$BASIC_DIR" "BASIC"
			run_category "$ADVANCED_DIR" "ADVANCED"
			run_category "$TRAITS_DIR" "TRAITS"
		fi
		;;
esac

if [ $RUN_CARGO_EXAMPLES -eq 1 ]; then
	run_all_cargo_examples
fi

if [ $RUN_STANDALONE -eq 1 ]; then
	run_standalone_projects
fi

# Summary
echo -e "\n${YELLOW}═══════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Summary${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════${NC}"
echo -e "Total:  $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
	echo -e "\n${GREEN}🎉 All examples passed!${NC}"
	exit 0
else
	echo -e "\n${RED}❌ Some examples failed${NC}"
	exit 1
fi

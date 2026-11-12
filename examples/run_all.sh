#!/bin/bash

# Lust Language - Example Runner
# Runs example files through the lust binary

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Defaults
BINARY="${LUST_BIN:-}"
BACKEND="${LUST_BACKEND:-}"
FILTER_JIT=1
CATEGORY_FILTER=""

# Directories
EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXAMPLES_DIR/.." && pwd)"
BASIC_DIR="$EXAMPLES_DIR/basic"
ADVANCED_DIR="$EXAMPLES_DIR/advanced"
TRAITS_DIR="$EXAMPLES_DIR/traits"
EFFECTS_DIR="$EXAMPLES_DIR/effects"
PROGRAMS_DIR="$EXAMPLES_DIR/programs"

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
			echo "  --category, -c <name>   Run only specified category (BASIC, ADVANCED, TRAITS, EFFECTS, PROGRAMS, ALL)"
			echo "  --all                   Run all categories including effects and programs"
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

	TOTAL=$((TOTAL + 1))

	echo -e "\n${CYAN}[$category]${NC} Running: ${BLUE}$(basename "$file")${NC}"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

	if [ -n "$BACKEND" ]; then
		cmd_args=("$BINARY" "run" "$file" "--backend" "$BACKEND")
	else
		cmd_args=("$BINARY" "$file")
	fi

	local output
	output=$("${cmd_args[@]}" 2>&1) || exit_code=$?

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

# Main execution
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
	PROGRAMS)
		run_category "$PROGRAMS_DIR" "PROGRAMS"
		;;
	ALL)
		run_category "$BASIC_DIR" "BASIC"
		run_category "$ADVANCED_DIR" "ADVANCED"
		run_category "$TRAITS_DIR" "TRAITS"
		run_category "$EFFECTS_DIR" "EFFECTS"
		run_category "$PROGRAMS_DIR" "PROGRAMS"
		;;
	*)
		# Default: run the standard 34 examples across BASIC, ADVANCED, and TRAITS
		run_category "$BASIC_DIR" "BASIC"
		run_category "$ADVANCED_DIR" "ADVANCED"
		run_category "$TRAITS_DIR" "TRAITS"
		;;
esac

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

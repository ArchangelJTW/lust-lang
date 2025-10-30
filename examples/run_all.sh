#!/bin/bash

# Lust Language - Example Runner
# Runs all example files through the release binary

set -e # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Paths
BINARY="lust"
EXAMPLES_DIR="./examples"
BASIC_DIR="$EXAMPLES_DIR/basic"
ADVANCED_DIR="$EXAMPLES_DIR/advanced"
TRAITS_DIR="$EXAMPLES_DIR/traits"

# Counters
PASSED=0
FAILED=0
TOTAL=0

# Check if binary exists
# if [ ! -f "$BINARY" ]; then
# 	echo -e "${RED}Error: Release binary not found at $BINARY${NC}"
# 	echo -e "${YELLOW}Run 'cargo build --release' first${NC}"
# 	exit 1
# fi

# Function to run a single example
run_example() {
	local file=$1
	local category=$2

	TOTAL=$((TOTAL + 1))

	echo -e "\n${CYAN}[$category]${NC} Running: ${BLUE}$(basename "$file")${NC}"
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

	if "$BINARY" "$file" 2>&1; then
		echo -e "${GREEN}✓ Passed${NC}"
		PASSED=$((PASSED + 1))
		return 0
	else
		echo -e "${RED}✗ Failed${NC}"
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
		run_example "$file" "$category"
	done
}

# Main execution
echo -e "${CYAN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Lust Language - Example Test Runner    ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════╝${NC}"

# Run basic examples
run_category "$BASIC_DIR" "BASIC"

# Run advanced examples
run_category "$ADVANCED_DIR" "ADVANCED"

# Run advanced examples
run_category "$TRAITS_DIR" "TRAITS"

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

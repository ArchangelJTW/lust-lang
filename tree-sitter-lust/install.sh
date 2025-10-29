#!/usr/bin/env bash
set -e

echo "Installing tree-sitter-lust parser for Neovim..."

# Generate the parser
echo "Generating parser..."
tree-sitter generate

# Find nvim-treesitter installation
NVIM_TS_PATH="$HOME/.local/share/nvim/lazy/nvim-treesitter"

if [ ! -d "$NVIM_TS_PATH" ]; then
	echo "Error: nvim-treesitter not found at $NVIM_TS_PATH"
	echo "Please adjust NVIM_TS_PATH in this script to match your installation"
	exit 1
fi

# Create queries directory if it doesn't exist
QUERIES_DIR="$NVIM_TS_PATH/queries/lust"
mkdir -p "$QUERIES_DIR"

# Copy query files
echo "Copying query files to $QUERIES_DIR..."
cp -v queries/*.scm "$QUERIES_DIR/" 2>/dev/null || echo "No query files found to copy"

TS_USER_PATH="$HOME/Desktop/Code/tree-sitter/tree-sitter-lust/"

cp -r * "$TS_USER_PATH"

echo ""
echo "✓ Parser generated"
echo "✓ Queries copied to nvim-treesitter"
echo ""
echo "Next steps:"
echo "  1. Open Neovim"
echo "  2. Run :TSInstall lust"
echo "  3. Open a .lust file to see syntax highlighting!"

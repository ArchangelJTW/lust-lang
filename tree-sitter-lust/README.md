# tree-sitter-lust

Tree-sitter grammar for the Lust programming language - a strictly typed, Lua-syntax inspired language with JIT compilation.

## Features

- Full syntax highlighting for Lust language constructs
- Support for functions, structs, enums, traits, and impl blocks
- Pattern matching and control flow highlighting
- Type annotations and generic types
- Method calls with `:` syntax
- Comments (`--` and `#` style)
- All operators and literals

## Installation

### Prerequisites

```bash
npm install -g tree-sitter-cli
```

### Building the Parser

```bash
cd tree-sitter-lust
npm install
tree-sitter generate
```

### Testing

```bash
# Parse a file
tree-sitter parse ../examples/hello.lust

# Run tests (if you create test cases)
tree-sitter test
```

## Editor Integration

### Neovim

#### Using lazy.nvim (Recommended)

Add to your Neovim config:

```lua
{
  "nvim-treesitter/nvim-treesitter",
  build = ":TSUpdate",
  config = function()
    -- Add lust parser
    local parser_config = require("nvim-treesitter.parsers").get_parser_configs()
    parser_config.lust = {
      install_info = {
        url = "~/Desktop/Code/rs/lust-lang/tree-sitter-lust", -- local path
        files = {"src/parser.c", "src/scanner.c"},
        branch = "main",
        generate_requires_npm = false,
        requires_generate_from_grammar = false,
      },
      filetype = "lust",
    }

    require("nvim-treesitter.configs").setup({
      highlight = {
        enable = true,
      },
    })
  end,
}
```

#### Manual Installation

1. Create the parser directory:
```bash
mkdir -p ~/.local/share/nvim/site/pack/tree-sitter/start/tree-sitter-lust
```

2. Copy the parser files:
```bash
cp -r tree-sitter-lust/* ~/.local/share/nvim/site/pack/tree-sitter/start/tree-sitter-lust/
```

3. Add to your Neovim config (`init.lua`):
```lua
-- Register the parser
local parser_config = require("nvim-treesitter.parsers").get_parser_configs()
parser_config.lust = {
  install_info = {
    url = "~/.local/share/nvim/site/pack/tree-sitter/start/tree-sitter-lust",
    files = {"src/parser.c", "src/scanner.c"}
  },
  filetype = "lust",
}

-- Set up filetype detection
vim.filetype.add({
  extension = {
    lust = "lust",
  },
})
```

4. Install the parser:
```vim
:TSInstall lust
```

### VS Code

VS Code now has tree-sitter support via extensions. Here are the options:

#### Option 1: Using tree-sitter-vscode extension

1. Install a tree-sitter extension for VS Code (like "Tree-sitter syntax highlighting")
2. Add this to your VS Code settings.json:

```json
{
  "treesitter.grammars": [
    {
      "language": "lust",
      "path": "/full/path/to/tree-sitter-lust"
    }
  ]
}
```

#### Option 2: Create a VS Code Extension

You can create a full VS Code extension that includes this tree-sitter grammar. Here's a basic structure:

1. Create a new directory:
```bash
mkdir vscode-lust
cd vscode-lust
npm init -y
```

2. Create `package.json`:
```json
{
  "name": "lust-language-support",
  "displayName": "Lust Language Support",
  "description": "Syntax highlighting for Lust programming language",
  "version": "0.1.0",
  "engines": {
    "vscode": "^1.70.0"
  },
  "contributes": {
    "languages": [{
      "id": "lust",
      "extensions": [".lust"],
      "configuration": "./language-configuration.json"
    }],
    "grammars": [{
      "language": "lust",
      "scopeName": "source.lust",
      "path": "./syntaxes/lust.tmLanguage.json"
    }]
  }
}
```

3. You can generate a TextMate grammar from the tree-sitter grammar or use tree-sitter directly with newer VS Code extensions.

## File Structure

```
tree-sitter-lust/
├── grammar.js           # Grammar definition
├── package.json         # NPM package config
├── binding.gyp          # Node.js binding config
├── src/
│   ├── parser.c        # Generated parser (after tree-sitter generate)
│   ├── scanner.c       # Custom scanner (if needed)
│   └── tree_sitter/
├── queries/
│   ├── highlights.scm  # Syntax highlighting queries
│   ├── injections.scm  # (optional) Language injections
│   └── locals.scm      # (optional) Local scope tracking
└── test/
    └── corpus/         # Test cases (optional)
```

## Language Support

This grammar supports:

- **Declarations**: functions, structs, enums, traits, impl blocks
- **Types**: primitives (int, float, bool, string), generics (Array<T>, Map<K,V>), Option<T>, Result<T,E>
- **Expressions**: binary/unary ops, calls, method calls, field access, indexing
- **Statements**: if/then/else, while/do, for loops, match statements
- **Pattern matching**: enum patterns with data extraction
- **Literals**: numbers, strings, booleans, nil
- **Comments**: `--` and `#` style
- **Special syntax**: `:` for method calls, `:as<T>()` for type casting

## Development

### Adding Test Cases

Create test files in `test/corpus/`:

```
================================================================================
Function declaration
================================================================================

function add(a: int, b: int): int
    return a + b
end

--------------------------------------------------------------------------------

(source_file
  (function_declaration
    name: (identifier)
    parameters: (parameter_list
      (parameter name: (identifier) type: (primitive_type))
      (parameter name: (identifier) type: (primitive_type)))
    return_type: (primitive_type)
    (return_statement (binary_expression))))
```

Run tests:
```bash
tree-sitter test
```

### Debugging

Parse a file and see the syntax tree:
```bash
tree-sitter parse examples/hello.lust
```

## Contributing

Contributions welcome! Areas for improvement:

- More comprehensive test coverage
- Additional query files (locals.scm, injections.scm)
- Better error recovery in the parser
- VS Code extension packaging

## Resources

- [Tree-sitter documentation](https://tree-sitter.github.io/tree-sitter/)
- [Lust language repository](https://github.com/yourusername/lust-lang)
- [Tree-sitter grammar development guide](https://tree-sitter.github.io/tree-sitter/creating-parsers)

## License

MIT

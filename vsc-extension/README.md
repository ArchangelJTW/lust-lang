# Lust Analyzer VS Code Extension

This extension wires the `lust-analyzer` Language Server into Visual Studio Code, providing real-time diagnostics for Lust source files.

## Features

- Full syntax highlighting matching the latest Tree-sitter grammar definitions.
- Automatic bracket matching, comments toggling, and auto-closing pairs.
- Real-time compiler and type-checker diagnostics as you type via `lust-analyzer`.
- Go-to-definition, hover documentation, completions, and inlay hints.
- Automatic detection of `lust-analyzer` in workspace targets, `~/.cargo/bin`, or system `PATH`.

## Requirements

- Build the language server once: `cargo build -p lust-analyzer`
- VS Code 1.105.0 or newer

## Extension Settings

`lustAnalyzer.serverPath` – absolute path to the `lust-analyzer` executable. When left blank the extension will look for `target/{debug,release}/lust-analyzer(.exe)` relative to the extension directory.

## Getting Started

1. Build the LSP once: `cargo build -p lust-analyzer`
2. Run `npm install`
3. Start the extension in VS Code via `Run Extension` (from the Run view) or package it with `npm run package`

Open any `.lust` file and diagnostics should appear automatically.

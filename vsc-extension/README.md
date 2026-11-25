# Lust Analyzer VS Code Extension

This extension wires the `lust-analyzer` Language Server into Visual Studio Code, providing syntax highlighting, indentation rules, and real-time diagnostics for Lust v3.5.9 (Waves 1–18) source files.

## Features

- Full Lust v3.5.9 (Waves 1–18) TextMate syntax highlighting (`syntaxes/lust.tmLanguage.json`), covering algebraic effects (`effect`/`handle`/`with`/`perform`/`resume`), structured resources (`defer`/`errdefer`/`scoped`), schemas (`schema`, `@` constraints, `typeof`, `Infer<T>`), `regex!"..."` literals, `@std/...` imports, test suites (`test`/`suite`/`describe`), numeric radices (`0x`/`0b`/`0o` with `_` separators), and full bitwise/pipe/compound operators.
- Automatic bracket matching, comment toggling, auto-closing pairs, and block indentation rules (`language-configuration.json`).
- Real-time compiler and type-checker diagnostics as you type via `lust-analyzer` (v0.7.8+).
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
3. Run grammar verification: `npm run test:grammar` (no VS Code host required)
4. Start the extension in VS Code via `Run Extension` (from the Run view) or package it with `npm run package`

Open any `.lust` file and diagnostics should appear automatically.

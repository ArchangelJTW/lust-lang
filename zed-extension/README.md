# Lust for Zed

Official [Zed](https://zed.dev) language extension for the **Lust** programming language (`.lust`), powered by the [`tree-sitter-lust`](../tree-sitter-lust) grammar.

## Features

- Syntax highlighting (`languages/lust/highlights.scm`) using standard Zed capture scopes
- Bracket matching & keyword-block pairing (`languages/lust/brackets.scm`)
- Syntax-aware auto-indentation (`languages/lust/indents.scm`)
- Code outline / symbol navigation for modules, structs, fields, enums, variants, traits, impl blocks, functions, methods, and extern items (`languages/lust/outline.scm`)
- Syntax scope overrides for strings and comments so auto-closing pairs respect string/comment boundaries (`languages/lust/overrides.scm`)
- Line comments (`-- `), block comments (`--[[ ... ]]`), auto-closing pairs, and 4-space indentation defaults (`languages/lust/config.toml`)

## Installing as a Dev Extension in Zed

1. Open Zed.
2. Open the Command Palette (`Cmd+Shift+P` on macOS / `Ctrl+Shift+P` on Linux).
3. Run **`zed: install dev extension`**.
4. Select this `zed-extension/` directory.
5. Open any `.lust` file — Zed will compile the `tree-sitter-lust` grammar and activate highlighting, outline, brackets, and auto-indentation.

> **Local / offline grammar development:** By default, `[grammars.lust]` in [`extension.toml`](extension.toml) references the Git repository at a pinned commit `rev` with `path = "tree-sitter-lust"`. If you are testing uncommitted or unpushed local grammar edits, temporarily point `repository` in `extension.toml` to a local `file://` URL (for example `repository = "file:///Users/you/work/lust-lang/repos/lust-lang-fork"`) and set `rev` to your local commit SHA, then run **`zed: rebuild dev extension`**.

## Syncing Grammar & Queries

Whenever [`tree-sitter-lust/grammar.js`](../tree-sitter-lust/grammar.js) is updated:

1. Regenerate and test the parser in `tree-sitter-lust/`:
   ```bash
   cd tree-sitter-lust
   tree-sitter generate
   tree-sitter test
   ```
2. Update `rev` in `zed-extension/extension.toml` to the commit containing the updated grammar.
3. Update `zed-extension/languages/lust/{highlights,brackets,indents,outline,overrides}.scm` if new node types or fields were added in `tree-sitter-lust/src/node-types.json`.
4. Validate all TOML configs and Tree-sitter queries empirically with the repository integration test:
   ```bash
   cargo test --offline --locked --test ng_w18_zed_highlighting
   ```

## Grammar Coverage Notes

All `.scm` queries are strictly grounded in the current `tree-sitter-lust/src/node-types.json` so Zed loads every query with zero errors.

- **Supported directly by `tree-sitter-lust`**: functions, lambdas, methods (`Type:method`), structs, enums, traits, impls, generics (`<T: Bound>`), modules, `use` imports, `extern` blocks, control flow (`if`/`elseif`/`else`/`while`/`for`/`do`/`return`/`break`/`continue`), `is`/`as`/`?` expressions, patterns, arrays, maps, tuples, numbers (hex/bin/oct/float; numeric underscores are present in `grammar.js` and take effect after running `tree-sitter generate`), strings, booleans, `--` / `#` / `--[[ ... ]]` comments, and `---` doc comments.
- **Modern Lust constructs not yet modeled as grammar rules in `tree-sitter-lust`**: `schema`, `service`, `provide`, `handle`, `with`, `effect`, `perform`, `resume`, `defer`, `errdefer`, `scoped`, pipe operator `|>`, and `&mut`. Reserved lexer keywords (`defer`, `errdefer`, `scoped`) are highlighted via identifier-match rules in `highlights.scm`; full AST support will be unlocked once added to `tree-sitter-lust/grammar.js`.

## Future LSP Integration (`lust-analyzer`)

This repository includes the Lust Language Server in [`crates/lust-analyzer`](../crates/lust-analyzer) (binary name: `lust-analyzer`). Zed requires a compiled WebAssembly extension module (`zed_extension_api` implementing `language_server_command`) to spawn custom LSP binaries; `[language_servers.lust-analyzer]` is documented in `extension.toml` and ready to be enabled once a Wasm crate is wired up.

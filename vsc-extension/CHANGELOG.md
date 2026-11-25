# Change Log

All notable changes to the "lust-analyzer" extension will be documented in this file.

Check [Keep a Changelog](http://keepachangelog.com/) for recommendations on how to structure this file.

## [0.7.8]

- Synchronized VS Code extension metadata and grammar with **Lust v3.5.9** (Waves 1–18) and `lust-analyzer` 0.7.8.
- Updated `syntaxes/lust.tmLanguage.json` with full Waves 1–18 syntax coverage:
  - Resource & algebraic effect keywords (`defer`, `errdefer`, `scoped`, `effect`, `handle`, `with`, `perform`, `resume`, `pure`).
  - Contextual declarations (`schema`, `type`, `service`, `effect`, `impl ... for ...`, `test`, `suite`, `describe`, `before_each`, `after_each`, `before_all`, `after_all`, `import ... from`).
  - Storage & visibility modifiers (`pub`, `priv`, `weak`, `ref`, `local`, `let`, `const`, `mut`, `module`, `extern`, `use`, `export`) and type operators (`typeof`, `is`, `as`).
  - `@` attributes/constraints (`@min`, `@max`, `@email`, `@derive`, `@test`, etc.) and `@std/...` module paths (`@std/sync`, `@std/json`, `@std/regex`, etc.).
  - Octal (`0o`/`0O`), hex (`0x`), binary (`0b`), underscore-separated integers/floats, scientific floats, and `regex!"..."` literals.
  - Full operator set (`|>`, `->`, `=>`, `::`, `//`, `//=`, `<<`, `>>`, `^^`, `~`, `!`, `?`, and all compound assignments `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `^=`, `&=`, `|=`, `^^=`, `<<=`, `>>=`).
  - Exact 3-dash doc comment matching (`---` vs `----` ordinary comments) with `@tag` highlighting.
  - Expanded built-in types, constructors (`Some`, `None`, `Ok`, `Err`), built-in functions, and stdlib namespaces; fixed single-letter type parameters (`T`, `K`, `V`) vs `ALL_CAPS` constants (min 2 chars).
- Updated `language-configuration.json` `indentationRules` for Waves 5–18 block constructs (`schema`, `scoped`, `effect`, `handle`, `with`, `match`, `test`, `suite`, `describe`, hooks, and `do`/`then`/`with`/`{` blocks).
- Added automated grammar and indentation verification (`test-grammar.mjs` and `tests/ng_w18_vsc_extension.rs`).

## [0.5.1]

- Added complete syntax highlighting grammar (`syntaxes/lust.tmLanguage.json`) matching latest Tree-sitter definitions.
- Added `language-configuration.json` for comments, brackets, and auto-closing pairs.
- Improved `lust-analyzer` binary resolution to check workspace targets, `~/.cargo/bin`, and `PATH`.
- Supported LSP capabilities: diagnostics, hovers, definitions, completions, inlay hints, and semantic tokens.

## [0.0.1]

- Initial release

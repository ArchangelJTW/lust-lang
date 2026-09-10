# Change Log

All notable changes to the "lust-analyzer" extension will be documented in this file.

Check [Keep a Changelog](http://keepachangelog.com/) for recommendations on how to structure this file.

## [0.5.1]

- Added complete syntax highlighting grammar (`syntaxes/lust.tmLanguage.json`) matching latest Tree-sitter definitions.
- Added `language-configuration.json` for comments, brackets, and auto-closing pairs.
- Improved `lust-analyzer` binary resolution to check workspace targets, `~/.cargo/bin`, and `PATH`.
- Supported LSP capabilities: diagnostics, hovers, definitions, completions, inlay hints, and semantic tokens.

## [0.0.1]

- Initial release
# Lust Analyzer

[lust-lang.dev](https://lust-lang.dev) · [Docs](https://lust-lang.dev/docs) · [Core crate](https://crates.io/crates/lust-lang)

Heavily WIP language server for the Lust scripting language. It targets editor integrations by forwarding parsing, typing, and diagnostics through the main `lust` library.

## Status
- Basic parsing + type checking go through, but large areas (completions, refactors, multi-file project awareness) are experimental or missing.
- Expect frequent breaking changes while the protocol surface settles.

## Trying It
```bash
git clone https://github.com/<your-org>/lust-lang
cargo run -p lust-analyzer
```

Point your editor’s LSP client at the spawned process to explore the current feature set. Contributions and bug reports are welcome!

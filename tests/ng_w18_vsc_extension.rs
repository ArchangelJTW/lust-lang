#![cfg(feature = "std")]

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn vsc_ext_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("vsc-extension")
}

fn collect_all_regex_strings(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                if matches!(k.as_str(), "match" | "begin" | "end") {
                    if let Some(s) = v.as_str() {
                        out.push(s.to_string());
                    }
                } else {
                    collect_all_regex_strings(v, out);
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                collect_all_regex_strings(item, out);
            }
        }
        _ => {}
    }
}

#[test]
fn test_vsc_extension_manifests_valid_and_synced() {
    let pkg_raw = fs::read_to_string(vsc_ext_dir().join("package.json"))
        .expect("Failed to read vsc-extension/package.json");
    let pkg: serde_json::Value =
        serde_json::from_str(&pkg_raw).expect("Invalid JSON in vsc-extension/package.json");

    assert_eq!(pkg["name"].as_str(), Some("lust-analyzer"));
    assert_eq!(pkg["version"].as_str(), Some("0.7.8"));
    assert_eq!(pkg["contributes"]["languages"][0]["id"].as_str(), Some("lust"));
    assert_eq!(
        pkg["contributes"]["grammars"][0]["scopeName"].as_str(),
        Some("source.lust")
    );
    assert_eq!(
        pkg["contributes"]["grammars"][0]["path"].as_str(),
        Some("./syntaxes/lust.tmLanguage.json")
    );

    let cfg_raw = fs::read_to_string(vsc_ext_dir().join("language-configuration.json"))
        .expect("Failed to read vsc-extension/language-configuration.json");
    let cfg: serde_json::Value = serde_json::from_str(&cfg_raw)
        .expect("Invalid JSON in vsc-extension/language-configuration.json");

    assert_eq!(cfg["comments"]["lineComment"].as_str(), Some("--"));
    assert!(cfg["indentationRules"]["increaseIndentPattern"]
        .as_str()
        .is_some_and(|p| p.contains("schema")
            && p.contains("scoped")
            && p.contains("handle")
            && p.contains("effect")
            && p.contains("test")));
    assert!(cfg["indentationRules"]["decreaseIndentPattern"]
        .as_str()
        .is_some_and(|p| p.contains("end") && p.contains("with")));
}

#[test]
fn test_vsc_tmgrammar_covers_all_lust_lexer_keywords_and_operators() {
    let grammar_raw =
        fs::read_to_string(vsc_ext_dir().join("syntaxes/lust.tmLanguage.json"))
            .expect("Failed to read lust.tmLanguage.json");
    let grammar: serde_json::Value =
        serde_json::from_str(&grammar_raw).expect("Invalid JSON in lust.tmLanguage.json");

    assert_eq!(grammar["name"].as_str(), Some("Lust"));
    assert_eq!(grammar["scopeName"].as_str(), Some("source.lust"));

    let repo = grammar["repository"]
        .as_object()
        .expect("repository must be an object");
    let repo_keys: HashSet<&str> = repo.keys().map(String::as_str).collect();
    for entry in grammar["patterns"]
        .as_array()
        .expect("patterns must be an array")
    {
        if let Some(inc) = entry.get("include").and_then(|v| v.as_str()) {
            let key = inc
                .strip_prefix('#')
                .expect("include must start with '#'");
            assert!(
                repo_keys.contains(key),
                "Missing repository key '{}' referenced by root patterns",
                key
            );
        }
    }

    let mut all_patterns = Vec::new();
    collect_all_regex_strings(&grammar, &mut all_patterns);
    let joined = all_patterns.join("\n");

    // Every keyword recognized by lust::lexer::TokenKind::keyword plus effect keywords
    let required_keywords = [
        "local", "function", "return", "if", "then", "else", "elseif", "end", "while", "do",
        "for", "in", "break", "continue", "struct", "enum", "trait", "impl", "as", "is", "true",
        "false", "and", "or", "not", "extern", "use", "module", "defer", "errdefer", "scoped",
        "effect", "handle", "with", "perform", "resume", "pure", "schema", "type", "typeof",
        "test", "pub", "priv", "weak", "ref", "const", "match",
    ];
    for kw in required_keywords {
        assert!(
            joined.contains(kw),
            "lust.tmLanguage.json missing keyword '{}'",
            kw
        );
    }

    // Verify all lexer TokenKind::keyword entries are covered
    for probe in [
        "local", "function", "return", "if", "then", "else", "elseif", "end", "while", "do",
        "for", "in", "break", "continue", "struct", "enum", "trait", "impl", "as", "is", "true",
        "false", "and", "or", "not", "extern", "use", "module", "defer", "errdefer", "scoped",
    ] {
        assert!(
            lust::lexer::TokenKind::keyword(probe).is_some(),
            "Expected '{}' to be a lexer keyword",
            probe
        );
    }

    // Verify multi-char operators and octal/regex support exist in grammar patterns
    for needle in [
        "0[oO]", "regex", "@std/", "\\|>", "->", "=>", "::", "//=", "//", "<<=", ">>=", "\\^\\^=",
        "\\^\\^", "<<", ">>", "----", "---(?!-)", "[A-Z][A-Z0-9_]+",
    ] {
        assert!(
            joined.contains(needle),
            "lust.tmLanguage.json missing expected pattern fragment '{}'",
            needle
        );
    }
}

#[test]
fn test_vsc_grammar_node_oracle_passes() {
    let script = vsc_ext_dir().join("test-grammar.mjs");
    assert!(script.is_file(), "Missing {}", script.display());

    for runtime in ["node", "bun"] {
        match Command::new(runtime)
            .arg(&script)
            .current_dir(vsc_ext_dir())
            .output()
        {
            Ok(out) => {
                assert!(
                    out.status.success(),
                    "{} test-grammar.mjs failed:\nstdout:\n{}\nstderr:\n{}",
                    runtime,
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                return;
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => panic!("Failed to launch {} test-grammar.mjs: {}", runtime, err),
        }
    }

    println!(
        "SKIP: neither `node` nor `bun` runtime found in PATH; skipping vsc-extension/test-grammar.mjs"
    );
}

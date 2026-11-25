#![cfg(feature = "std")]

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use tree_sitter::{Parser, Point, Query, QueryCursor};

fn zed_ext_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("zed-extension")
}

fn lang_dir() -> PathBuf {
    zed_ext_dir().join("languages").join("lust")
}

fn make_parser() -> Parser {
    let mut parser = Parser::new();
    parser
        .set_language(tree_sitter_lust::language())
        .expect("Failed to load tree-sitter-lust language");
    parser
}

/// Official Zed theme syntax capture whitelist (docs/src/extensions/languages.md +
/// standard hierarchical theme scopes resolved by Zed's SyntaxTheme).
/// Deliberately excludes non-Zed scopes like `module`.
const ZED_SUPPORTED_HIGHLIGHT_CAPTURES: &[&str] = &[
    "attribute",
    "boolean",
    "comment",
    "comment.doc",
    "constant",
    "constant.builtin",
    "constructor",
    "embedded",
    "emphasis",
    "emphasis.strong",
    "enum",
    "function",
    "function.builtin",
    "function.method",
    "hint",
    "keyword",
    "label",
    "link_text",
    "link_uri",
    "namespace",
    "number",
    "operator",
    "predictive",
    "preproc",
    "primary",
    "property",
    "punctuation",
    "punctuation.bracket",
    "punctuation.delimiter",
    "punctuation.list_marker",
    "punctuation.special",
    "string",
    "string.escape",
    "string.regex",
    "string.special",
    "string.special.symbol",
    "tag",
    "tag.doctype",
    "text.literal",
    "title",
    "type",
    "type.builtin",
    "variable",
    "variable.parameter",
    "variable.special",
    "variant",
];

fn is_valid_zed_capture(name: &str) -> bool {
    ZED_SUPPORTED_HIGHLIGHT_CAPTURES.contains(&name)
}

fn collect_captures<'a>(
    query_src: &str,
    source: &'a str,
) -> (Vec<(String, &'a str)>, HashMap<String, Vec<&'a str>>) {
    let language = tree_sitter_lust::language();
    let query = Query::new(language, query_src)
        .unwrap_or_else(|e| panic!("Query compilation failed: {:?}", e));
    let mut parser = make_parser();
    let tree = parser
        .parse(source, None)
        .expect("tree-sitter-lust failed to parse sample source");

    let mut cursor = QueryCursor::new();
    let capture_names = query.capture_names();
    let mut ordered = Vec::new();
    let mut by_capture: HashMap<String, Vec<&'a str>> = HashMap::new();

    for m in cursor.matches(&query, tree.root_node(), source.as_bytes()) {
        for cap in m.captures {
            let name = capture_names[cap.index as usize].clone();
            let text = &source[cap.node.start_byte()..cap.node.end_byte()];
            ordered.push((name.clone(), text));
            by_capture.entry(name).or_default().push(text);
        }
    }

    (ordered, by_capture)
}

/// Captures produced in Tree-sitter `QueryCursor::captures` order, filtered by
/// valid Zed theme scopes (mirroring Zed's `BufferChunkHighlights` stack where
/// inner child nodes and later query patterns sit on top of outer/earlier ones).
#[derive(Debug, Clone)]
struct CaptureRegion {
    range: Range<usize>,
    capture_name: String,
}

fn flatten_capture_regions(query_src: &str, source: &str) -> Vec<CaptureRegion> {
    let language = tree_sitter_lust::language();
    let query = Query::new(language, query_src)
        .unwrap_or_else(|e| panic!("Query compilation failed: {:?}", e));
    let mut parser = make_parser();
    let tree = parser
        .parse(source, None)
        .expect("tree-sitter-lust failed to parse source");
    assert!(
        !tree.root_node().has_error(),
        "Source must parse without tree-sitter ERROR nodes:\n{}",
        tree.root_node().to_sexp()
    );

    let mut cursor = QueryCursor::new();
    let capture_names = query.capture_names();
    let mut regions = Vec::new();

    for (mat, cap_ix) in cursor.captures(&query, tree.root_node(), source.as_bytes()) {
        let cap = mat.captures[cap_ix];
        let name = &capture_names[cap.index as usize];
        if !is_valid_zed_capture(name) {
            continue;
        }
        regions.push(CaptureRegion {
            range: cap.node.start_byte()..cap.node.end_byte(),
            capture_name: name.clone(),
        });
    }

    regions
}

/// Resolves the winning Zed highlight capture for a specific byte span using
/// `.rev().find_map(...)` over tree-order captures (matching Zed's top-of-stack
/// highlight resolution).
fn winning_capture_for_span<'a>(
    regions: &'a [CaptureRegion],
    span: Range<usize>,
) -> Option<&'a str> {
    regions.iter().rev().find_map(|region| {
        if region.range.start <= span.start && region.range.end >= span.end {
            Some(region.capture_name.as_str())
        } else {
            None
        }
    })
}

/// Finds the byte span of `token` inside the unique substring `context` in `source`.
fn token_span_in(source: &str, context: &str, token: &str) -> Range<usize> {
    let ctx_start = source
        .find(context)
        .unwrap_or_else(|| panic!("Context {:?} not found in source", context));
    let rel_start = context
        .find(token)
        .unwrap_or_else(|| panic!("Token {:?} not found in context {:?}", token, context));
    let start = ctx_start + rel_start;
    start..(start + token.len())
}

/// Faithful simulation of Zed's `BufferSnapshot::suggest_autoindents`
/// (`crates/language/src/buffer.rs`) for `@indent`, `@start`, `@end`, and `@outdent`.
/// Returns the suggested indent depth (in tab stops) for every line in `source`.
fn simulate_zed_autoindents(indents_src: &str, source: &str) -> Vec<usize> {
    let language = tree_sitter_lust::language();
    let query = Query::new(language, indents_src)
        .unwrap_or_else(|e| panic!("indents.scm compilation failed: {:?}", e));
    let mut parser = make_parser();
    let tree = parser
        .parse(source, None)
        .expect("tree-sitter-lust failed to parse indent sample");
    assert!(
        !tree.root_node().has_error(),
        "Indent sample must parse cleanly:\n{}",
        tree.root_node().to_sexp()
    );

    let capture_names = query.capture_names();
    let indent_ix = capture_names.iter().position(|n| n == "indent").map(|i| i as u32);
    let start_ix = capture_names.iter().position(|n| n == "start").map(|i| i as u32);
    let end_ix = capture_names.iter().position(|n| n == "end").map(|i| i as u32);
    let outdent_ix = capture_names.iter().position(|n| n == "outdent").map(|i| i as u32);

    let mut indent_ranges: Vec<Range<Point>> = Vec::new();
    let mut outdent_positions: Vec<Point> = Vec::new();

    let mut cursor = QueryCursor::new();
    for mat in cursor.matches(&query, tree.root_node(), source.as_bytes()) {
        let mut start: Option<Point> = None;
        let mut end: Option<Point> = None;

        for cap in mat.captures {
            if Some(cap.index) == indent_ix {
                start.get_or_insert(cap.node.start_position());
                end.get_or_insert(cap.node.end_position());
            } else if Some(cap.index) == start_ix {
                start = Some(cap.node.end_position());
            } else if Some(cap.index) == end_ix {
                end = Some(cap.node.start_position());
            } else if Some(cap.index) == outdent_ix {
                outdent_positions.push(cap.node.start_position());
            }
        }

        if let Some((s, e)) = start.zip(end) {
            if s.row == e.row {
                continue;
            }
            let range = s..e;
            match indent_ranges.binary_search_by_key(&range.start, |r| r.start) {
                Err(ix) => indent_ranges.insert(ix, range),
                Ok(ix) => {
                    let prev_range = &mut indent_ranges[ix];
                    if range.end > prev_range.end {
                        prev_range.end = range.end;
                    }
                }
            }
        }
    }

    // Exact Zed @outdent truncation: find innermost containing indent range and truncate its end.
    outdent_positions.sort();
    for outdent_pos in outdent_positions {
        if let Some(range_to_truncate) = indent_ranges
            .iter_mut()
            .rfind(|indent_range| indent_range.contains(&outdent_pos))
        {
            range_to_truncate.end = outdent_pos;
        }
    }

    let lines: Vec<&str> = source.lines().collect();
    let line_indent_col = |row: usize| -> usize {
        lines[row]
            .bytes()
            .take_while(|b| *b == b' ' || *b == b'\t')
            .count()
    };

    let mut computed_depths = vec![0usize; lines.len()];
    let mut prev_non_blank_row: usize = 0;

    for row in 1..lines.len() {
        let prev_row = prev_non_blank_row;
        let prev_row_start = Point::new(prev_row, line_indent_col(prev_row));
        let row_start = Point::new(row, line_indent_col(row));

        let mut indent_from_prev_row = false;
        let mut outdent_to_row = usize::MAX;

        for range in &indent_ranges {
            if range.start.row >= row {
                break;
            }
            if range.start.row == prev_row && range.end > row_start {
                indent_from_prev_row = true;
            }
            if range.end > prev_row_start && range.end <= row_start {
                outdent_to_row = outdent_to_row.min(range.start.row);
            }
        }

        let (basis_row, delta) = if outdent_to_row == prev_row {
            (prev_row, Ordering::Equal)
        } else if indent_from_prev_row {
            (prev_row, Ordering::Greater)
        } else if outdent_to_row < prev_row {
            (outdent_to_row, Ordering::Equal)
        } else {
            (prev_row, Ordering::Equal)
        };

        computed_depths[row] = match delta {
            Ordering::Greater => computed_depths[basis_row] + 1,
            Ordering::Equal => computed_depths[basis_row],
            Ordering::Less => computed_depths[basis_row].saturating_sub(1),
        };

        if !lines[row].trim().is_empty() {
            prev_non_blank_row = row;
        }
    }

    computed_depths
}

#[test]
fn test_zed_extension_toml_valid_and_required_fields() {
    let ext_path = zed_ext_dir().join("extension.toml");
    let raw = fs::read_to_string(&ext_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", ext_path.display(), e));
    let doc: toml::Table = raw
        .parse()
        .unwrap_or_else(|e| panic!("Invalid TOML in {}: {}", ext_path.display(), e));

    assert_eq!(doc.get("id").and_then(|v| v.as_str()), Some("lust"));
    assert_eq!(doc.get("name").and_then(|v| v.as_str()), Some("Lust"));
    assert_eq!(
        doc.get("schema_version").and_then(|v| v.as_integer()),
        Some(1)
    );
    assert!(
        doc.get("version")
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.is_empty()),
        "extension.toml must have non-empty version"
    );
    assert!(
        doc.get("description")
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.is_empty()),
        "extension.toml must have non-empty description"
    );

    let grammar = doc
        .get("grammars")
        .and_then(|g| g.get("lust"))
        .expect("extension.toml must define [grammars.lust]");
    assert!(
        grammar
            .get("repository")
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.is_empty()),
        "[grammars.lust] must specify repository"
    );

    let rev = grammar
        .get("rev")
        .and_then(|v| v.as_str())
        .expect("[grammars.lust] must specify rev");
    assert_eq!(
        rev.len(),
        40,
        "[grammars.lust].rev must be a full 40-char commit SHA, got {:?}",
        rev
    );
    assert!(
        rev.bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
        "[grammars.lust].rev must be lowercase hex, got {:?}",
        rev
    );

    // Verify the pinned revision actually exists as a commit in this git repository.
    let output = Command::new("git")
        .args(["cat-file", "-t", rev])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute git cat-file");
    assert!(
        output.status.success(),
        "[grammars.lust].rev '{}' does not exist in git: {}",
        rev,
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "commit",
        "[grammars.lust].rev '{}' must resolve to a git commit",
        rev
    );

    assert_eq!(
        grammar.get("path").and_then(|v| v.as_str()),
        Some("tree-sitter-lust")
    );
}

#[test]
fn test_zed_language_config_toml_valid_and_required_fields() {
    let cfg_path = lang_dir().join("config.toml");
    let raw = fs::read_to_string(&cfg_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", cfg_path.display(), e));
    let doc: toml::Table = raw
        .parse()
        .unwrap_or_else(|e| panic!("Invalid TOML in {}: {}", cfg_path.display(), e));

    assert_eq!(doc.get("name").and_then(|v| v.as_str()), Some("Lust"));
    assert_eq!(doc.get("grammar").and_then(|v| v.as_str()), Some("lust"));

    let suffixes: Vec<&str> = doc
        .get("path_suffixes")
        .and_then(|v| v.as_array())
        .expect("config.toml must define path_suffixes")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(suffixes.contains(&"lust"));

    let line_comments: Vec<&str> = doc
        .get("line_comments")
        .and_then(|v| v.as_array())
        .expect("config.toml must define line_comments")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(line_comments.contains(&"-- "));

    let block_comment: Vec<&str> = doc
        .get("block_comment")
        .and_then(|v| v.as_array())
        .expect("config.toml must define block_comment")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(block_comment, vec!["--[[", "]]"]);

    assert_eq!(
        doc.get("autoclose_before").and_then(|v| v.as_str()),
        Some(";:.,=}])>")
    );
    assert_eq!(doc.get("tab_size").and_then(|v| v.as_integer()), Some(4));
    assert_eq!(doc.get("hard_tabs").and_then(|v| v.as_bool()), Some(false));

    let brackets = doc
        .get("brackets")
        .and_then(|v| v.as_array())
        .expect("config.toml must define brackets");
    let pairs: HashSet<(&str, &str)> = brackets
        .iter()
        .map(|b| {
            (
                b.get("start").and_then(|v| v.as_str()).unwrap(),
                b.get("end").and_then(|v| v.as_str()).unwrap(),
            )
        })
        .collect();
    for required in [("(", ")"), ("[", "]"), ("{", "}"), ("\"", "\""), ("'", "'")] {
        assert!(
            pairs.contains(&required),
            "config.toml brackets missing pair {:?}",
            required
        );
    }
}

#[test]
fn test_all_zed_scm_queries_compile_cleanly_and_use_valid_captures() {
    let language = tree_sitter_lust::language();
    let files = [
        lang_dir().join("highlights.scm"),
        lang_dir().join("brackets.scm"),
        lang_dir().join("indents.scm"),
        lang_dir().join("outline.scm"),
        lang_dir().join("overrides.scm"),
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tree-sitter-lust")
            .join("queries")
            .join("highlights.scm"),
    ];

    for path in &files {
        let src = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
        let query = Query::new(language, &src).unwrap_or_else(|e| {
            panic!(
                "Tree-sitter query error in {} at row {}, col {}: {:?}",
                path.display(),
                e.row + 1,
                e.column + 1,
                e
            )
        });
        assert!(
            query.pattern_count() > 0,
            "{} must contain at least one query pattern",
            path.display()
        );

        // Enforce Zed official supported highlight capture whitelist on Zed's highlights.scm.
        if path == &lang_dir().join("highlights.scm") {
            let unsupported: Vec<&str> = query
                .capture_names()
                .iter()
                .map(String::as_str)
                .filter(|cap| !is_valid_zed_capture(cap))
                .collect();
            assert!(
                unsupported.is_empty(),
                "zed-extension/languages/lust/highlights.scm contains unsupported Zed theme captures: {:?}",
                unsupported
            );
            assert!(
                !query.capture_names().iter().any(|c| c == "module"),
                "highlights.scm must use @namespace instead of unmapped @module"
            );
        }

        println!(
            "OK: {} ({} patterns, {} capture names)",
            path.strip_prefix(env!("CARGO_MANIFEST_DIR"))
                .unwrap_or(path)
                .display(),
            query.pattern_count(),
            query.capture_names().len()
        );
    }
}

#[test]
fn test_zed_overrides_scm_matches_strings_and_comments() {
    let overrides_src = fs::read_to_string(lang_dir().join("overrides.scm")).unwrap();
    let sample = r#"
--- Doc comment
-- Line comment
# Hash comment
--[[ Block comment ]]
local a = "double quoted"
local b = 'single quoted'
"#;
    let (_ord, by_cap) = collect_captures(&overrides_src, sample);

    assert!(
        by_cap
            .get("string")
            .is_some_and(|v| v.contains(&"\"double quoted\"") && v.contains(&"'single quoted'")),
        "overrides.scm must capture strings as @string; got {:?}",
        by_cap
    );
    let comments = by_cap
        .get("comment.inclusive")
        .expect("overrides.scm must capture comments as @comment.inclusive");
    assert!(comments.iter().any(|c| c.starts_with("--- Doc comment")));
    assert!(comments.iter().any(|c| c.starts_with("-- Line comment")));
    assert!(comments.iter().any(|c| c.starts_with("# Hash comment")));
    assert!(comments.iter().any(|c| c.starts_with("--[[ Block comment")));
}

#[test]
fn test_zed_highlights_winning_captures_oracle() {
    let highlights_src = fs::read_to_string(lang_dir().join("highlights.scm")).unwrap();
    let sample = r#"
--- Module doc comment
module math.geometry

use std.io as Io
use std.collections.{List}

-- Regular line comment

extern "C"
    function libc.puts(s: string): int
    const C.MAX_BUF: int
end

function Point.new(x: int, y: int): Option<int>
    local r: Result<int, string> = nil
    local arr: Array = nil
    local dict: Map<string, int> = nil
    local p: foo.Bar = nil
    local m = { foo = 1, [k] = 2, [MAX_KEY] = 3 }
    local s = geom.Point { x = 1 }
    if self is Shape.Circle(rad) then
        print("hit")
    end
    defer
    errdefer
    scoped
    return nil
end

function a.b.helper(): int
    return 0
end
"#;

    let regions = flatten_capture_regions(&highlights_src, sample);

    let assertions: &[(&str, &str, &str)] = &[
        // Built-in generic/collection types must win over @type
        ("Option<int>", "Option", "type.builtin"),
        ("Result<int, string>", "Result", "type.builtin"),
        ("local arr: Array", "Array", "type.builtin"),
        ("Map<string, int>", "Map", "type.builtin"),
        // Module & import namespaces
        ("module math.geometry", "math", "namespace"),
        ("module math.geometry", "geometry", "namespace"),
        ("use std.io as Io", "std", "namespace"),
        ("use std.io as Io", "io", "namespace"),
        ("use std.io as Io", "Io", "type"),
        ("use std.collections.{List}", "collections", "namespace"),
        ("use std.collections.{List}", "List", "namespace"),
        // Scoped function declarations (2-segment and 3-segment)
        ("function Point.new(", "Point", "type"),
        ("function Point.new(", "new", "function"),
        ("function a.b.helper()", "a", "namespace"),
        ("function a.b.helper()", "b", "namespace"),
        ("function a.b.helper()", "helper", "function"),
        // Scoped extern items
        ("function libc.puts(", "libc", "namespace"),
        ("function libc.puts(", "puts", "function"),
        ("const C.MAX_BUF:", "C", "namespace"),
        ("const C.MAX_BUF:", "MAX_BUF", "constant"),
        // Scoped enum pattern & struct literal
        ("Shape.Circle(rad)", "Shape", "type"),
        ("Shape.Circle(rad)", "Circle", "constructor"),
        ("geom.Point { x = 1 }", "geom", "namespace"),
        ("geom.Point { x = 1 }", "Point", "constructor"),
        ("geom.Point { x = 1 }", "x", "property"),
        // Scoped type annotation
        ("local p: foo.Bar", "foo", "namespace"),
        ("local p: foo.Bar", "Bar", "type"),
        // Map keys: bare identifier is @property, computed identifier stays @variable / @constant
        ("{ foo = 1, [k] = 2, [MAX_KEY] = 3 }", "foo", "property"),
        ("{ foo = 1, [k] = 2, [MAX_KEY] = 3 }", "k", "variable"),
        ("{ foo = 1, [k] = 2, [MAX_KEY] = 3 }", "MAX_KEY", "constant"),
        // Special variables, builtins, reserved lexer keywords, comments
        ("if self is Shape.Circle", "self", "variable.special"),
        ("return nil", "nil", "constant.builtin"),
        ("print(\"hit\")", "print", "function.builtin"),
        ("    defer\n", "defer", "keyword"),
        ("    errdefer\n", "errdefer", "keyword"),
        ("    scoped\n", "scoped", "keyword"),
        ("--- Module doc comment", "--- Module doc comment", "comment.doc"),
        ("-- Regular line comment", "-- Regular line comment", "comment"),
    ];

    for &(context, token, expected) in assertions {
        let span = token_span_in(sample, context, token);
        let actual = winning_capture_for_span(&regions, span.clone());
        assert_eq!(
            actual,
            Some(expected),
            "Winning Zed highlight mismatch for token {:?} in {:?} (bytes {:?}): expected @{}, got {:?}",
            token,
            context,
            span,
            expected,
            actual
        );
    }
}

#[test]
fn test_zed_indents_suggest_autoindents_oracle() {
    let indents_src = fs::read_to_string(lang_dir().join("indents.scm")).unwrap();

    // Case 1: Function containing `if ... elseif ... elseif ... else ... end` + trailing statement
    let src_multi_branch = "\
function classify(a: int, b: int, c: int): int
    if a > 0 then
        x = 1
    elseif b > 0 then
        y = 2
    elseif c > 0 then
        w = 4
    else
        z = 3
    end
    return x
end";
    let expected_multi_branch = vec![
        0, // function classify(...)
        1, //     if a > 0 then
        2, //         x = 1
        1, //     elseif b > 0 then
        2, //         y = 2
        1, //     elseif c > 0 then
        2, //         w = 4
        1, //     else
        2, //         z = 3
        1, //     end
        1, //     return x  (enclosing function range must NOT be truncated!)
        0, // end
    ];
    let actual_multi_branch = simulate_zed_autoindents(&indents_src, src_multi_branch);
    assert_eq!(
        actual_multi_branch, expected_multi_branch,
        "Auto-indent mismatch on multi-branch if/elseif/elseif/else inside function"
    );

    // Case 2: Nested `if/else/end` inside an `elseif` branch inside a function
    let src_nested = "\
function nested(a: bool, b: bool, inner: bool): int
    if a then
        x = 1
    elseif b then
        if inner then
            u = 1
        else
            v = 2
        end
    else
        z = 3
    end
    return 0
end";
    let expected_nested = vec![
        0, // function nested(...)
        1, //     if a then
        2, //         x = 1
        1, //     elseif b then
        2, //         if inner then
        3, //             u = 1
        2, //         else
        3, //             v = 2
        2, //         end
        1, //     else
        2, //         z = 3
        1, //     end
        1, //     return 0
        0, // end
    ];
    let actual_nested = simulate_zed_autoindents(&indents_src, src_nested);
    assert_eq!(
        actual_nested, expected_nested,
        "Auto-indent mismatch on nested if/else inside elseif branch"
    );

    // Case 3: Empty `elseif` body (cursor on blank line after hitting Enter on `elseif b then`)
    let src_empty_elseif = "\
function empty_branch(a: bool, b: bool)
    if a then
        x = 1
    elseif b then

    else
        z = 3
    end
    return x
end";
    let expected_empty_elseif = vec![
        0, // function empty_branch(...)
        1, //     if a then
        2, //         x = 1
        1, //     elseif b then
        2, //         <blank line inside empty elseif body>
        1, //     else
        2, //         z = 3
        1, //     end
        1, //     return x
        0, // end
    ];
    let actual_empty_elseif = simulate_zed_autoindents(&indents_src, src_empty_elseif);
    assert_eq!(
        actual_empty_elseif, expected_empty_elseif,
        "Auto-indent mismatch on blank line inside empty elseif body"
    );
}

#[test]
fn test_zed_highlights_and_outline_on_representative_lust_program() {
    let sample = r#"
--- Module documentation for geometry utilities
module math.geometry

use std.io.{println}

-- Line comment
--[[ Block comment across
     multiple lines ]]

struct Point<T>
    x: T,
    y: T,
end

enum Shape
    Circle(float),
    Rect(float, float),
end

trait Area
    function area(self): float
end

impl Area for Shape
    function area(self): float
        if self is Shape.Circle(r) then
            return 3.14159 * r * r
        else
            return 0.0
        end
    end
end

local function compute(limit: int, label: string): int
    local total = 0
    local active: bool = true
    local items = [1, 2, 3]
    for i = 1, limit do
        if i == 2 and active then
            total += i
        elseif i > 10 then
            break
        end
    end
    local p = Point { x = 1, y = 2 }
    println("point=" .. label .. tostring(p.x))
    return total
end
"#;

    // 1. Verify sample parses cleanly without syntax errors.
    let mut parser = make_parser();
    let tree = parser.parse(sample, None).expect("Failed to parse sample");
    assert!(
        !tree.root_node().has_error(),
        "Representative Lust program must parse without tree-sitter ERROR nodes:\n{}",
        tree.root_node().to_sexp()
    );

    // 2. Verify highlights.scm captures.
    let highlights_src = fs::read_to_string(lang_dir().join("highlights.scm")).unwrap();
    let (_ordered, by_cap) = collect_captures(&highlights_src, sample);

    for required_cap in [
        "keyword",
        "function",
        "function.method",
        "function.builtin",
        "type",
        "type.builtin",
        "variable",
        "variable.special",
        "variable.parameter",
        "property",
        "constructor",
        "number",
        "string",
        "boolean",
        "operator",
        "punctuation.bracket",
        "punctuation.delimiter",
        "comment",
        "comment.doc",
        "namespace",
    ] {
        assert!(
            by_cap.contains_key(required_cap),
            "Expected capture @{} to fire on representative Lust program; got keys: {:?}",
            required_cap,
            by_cap.keys().collect::<Vec<_>>()
        );
    }

    assert!(by_cap["keyword"].contains(&"function"));
    assert!(by_cap["keyword"].contains(&"struct"));
    assert!(by_cap["keyword"].contains(&"enum"));
    assert!(by_cap["keyword"].contains(&"trait"));
    assert!(by_cap["keyword"].contains(&"impl"));
    assert!(by_cap["keyword"].contains(&"if"));
    assert!(by_cap["keyword"].contains(&"return"));
    assert!(by_cap["keyword"].contains(&"break"));
    assert!(by_cap["keyword"].contains(&"end"));

    assert!(by_cap["function"].contains(&"compute"));
    assert!(by_cap["function.method"].contains(&"area"));
    assert!(by_cap["function.builtin"].contains(&"println"));
    assert!(by_cap["function.builtin"].contains(&"tostring"));

    assert!(by_cap["namespace"].contains(&"math"));
    assert!(by_cap["namespace"].contains(&"geometry"));
    assert!(by_cap["namespace"].contains(&"std"));
    assert!(by_cap["namespace"].contains(&"io"));

    assert!(by_cap["type"].contains(&"Point"));
    assert!(by_cap["type"].contains(&"Shape"));
    assert!(by_cap["type"].contains(&"Area"));
    assert!(by_cap["type.builtin"].contains(&"int"));
    assert!(by_cap["type.builtin"].contains(&"float"));
    assert!(by_cap["type.builtin"].contains(&"string"));
    assert!(by_cap["type.builtin"].contains(&"bool"));

    assert!(by_cap["variable.special"].contains(&"self"));
    assert!(by_cap["variable.parameter"].contains(&"limit"));
    assert!(by_cap["property"].contains(&"x"));
    assert!(by_cap["property"].contains(&"y"));
    assert!(by_cap["constructor"].contains(&"Circle"));
    assert!(by_cap["constructor"].contains(&"Rect"));
    assert!(by_cap["constructor"].contains(&"Point"));

    assert!(by_cap["number"].contains(&"3.14159"));
    assert!(by_cap["number"].contains(&"10"));
    assert!(by_cap["boolean"].contains(&"true"));
    assert!(by_cap["operator"].contains(&"+="));
    assert!(by_cap["operator"].contains(&".."));
    assert!(by_cap["operator"].contains(&"and"));
    assert!(by_cap["comment"]
        .iter()
        .any(|c| c.starts_with("-- Line comment")));
    assert!(by_cap["comment"]
        .iter()
        .any(|c| c.starts_with("--[[ Block comment")));
    assert!(by_cap["comment.doc"]
        .iter()
        .any(|c| c.starts_with("--- Module documentation")));

    // 3. Verify string literal highlighting separately as well.
    let str_sample = r#"local greeting = "hello, \"lust\""
local single = 'world'
local MAX_ITEMS = 42
local empty = nil
"#;
    let (_str_ord, str_caps) = collect_captures(&highlights_src, str_sample);
    assert!(str_caps["string"].contains(&"\"hello, \\\"lust\\\"\""));
    assert!(str_caps["string"].contains(&"'world'"));
    assert!(str_caps["constant"].contains(&"MAX_ITEMS"));
    assert!(str_caps["constant.builtin"].contains(&"nil"));

    // 4. Verify outline.scm captures (@item, @name, @context, @annotation).
    let outline_src = fs::read_to_string(lang_dir().join("outline.scm")).unwrap();
    let (_out_ord, out_caps) = collect_captures(&outline_src, sample);

    assert!(out_caps.contains_key("item"), "outline.scm must capture @item");
    assert!(out_caps.contains_key("name"), "outline.scm must capture @name");
    assert!(
        out_caps.contains_key("annotation"),
        "outline.scm must capture @annotation for doc comments"
    );

    let names: HashSet<&str> = out_caps["name"].iter().copied().collect();
    for expected_symbol in [
        "math.geometry",
        "Point",
        "x",
        "y",
        "Shape",
        "Circle",
        "Rect",
        "Area",
        "area",
        "compute",
    ] {
        assert!(
            names.contains(expected_symbol),
            "outline.scm @name missing expected symbol '{}'; got: {:?}",
            expected_symbol,
            names
        );
    }

    // 5. Verify brackets.scm and indents.scm fire on the representative program.
    let brackets_src = fs::read_to_string(lang_dir().join("brackets.scm")).unwrap();
    let (_br_ord, br_caps) = collect_captures(&brackets_src, sample);
    assert!(br_caps["open"].contains(&"("));
    assert!(br_caps["close"].contains(&")"));
    assert!(br_caps["open"].contains(&"["));
    assert!(br_caps["close"].contains(&"]"));
    assert!(br_caps["open"].contains(&"{"));
    assert!(br_caps["close"].contains(&"}"));
    assert!(br_caps["open"].contains(&"function"));
    assert!(br_caps["close"].contains(&"end"));

    let indents_src = fs::read_to_string(lang_dir().join("indents.scm")).unwrap();
    let (_in_ord, in_caps) = collect_captures(&indents_src, sample);
    assert!(!in_caps["indent"].is_empty());
    assert!(in_caps["start"].contains(&"then"));
    assert!(in_caps["start"].contains(&"else"));
    assert!(in_caps["end"].contains(&"end"));
    assert!(in_caps["outdent"].contains(&"elseif"));
    assert!(in_caps["outdent"].contains(&"else"));
}

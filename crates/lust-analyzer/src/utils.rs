use lust::ast::{Type, TypeKind};
use lust::{LustConfig, Span};
use tower_lsp::lsp_types::{Position, Range};
pub(crate) fn compute_line_offsets(text: &str) -> Vec<usize> {
    let mut offsets = Vec::new();
    offsets.push(0);
    for (idx, ch) in text.char_indices() {
        if ch == '\n' {
            offsets.push(idx + ch.len_utf8());
        }
    }

    offsets
}

pub(crate) fn nth_char_byte_index(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map(|(idx, _)| idx)
        .unwrap_or_else(|| text.len())
}

pub(crate) fn position_to_offset(
    text: &str,
    position: Position,
    line_offsets: &[usize],
) -> Option<usize> {
    let line_idx = position.line as usize;
    if line_idx >= line_offsets.len() {
        return None;
    }

    let line_start = line_offsets[line_idx];
    let line_end = line_offsets
        .get(line_idx + 1)
        .copied()
        .unwrap_or_else(|| text.len());
    if line_start > line_end || line_end > text.len() {
        return None;
    }

    let line_text = &text[line_start..line_end];
    let char_index = position.character as usize;
    let byte_in_line = nth_char_byte_index(line_text, char_index);
    Some(line_start + byte_in_line.min(line_text.len()))
}

pub(crate) fn offset_to_position(text: &str, offset: usize, line_offsets: &[usize]) -> Position {
    let clamped_offset = offset.min(text.len());
    let mut line_idx = 0;
    while line_idx + 1 < line_offsets.len() && line_offsets[line_idx + 1] <= clamped_offset {
        line_idx += 1;
    }

    let line_start = line_offsets.get(line_idx).copied().unwrap_or(0);
    let char_count = text[line_start..clamped_offset].chars().count();
    Position::new(line_idx as u32, char_count as u32)
}

pub(crate) fn prev_char_index(text: &str, mut offset: usize) -> Option<(usize, char)> {
    if offset == 0 {
        return None;
    }

    if offset > text.len() {
        offset = text.len();
    }

    let slice = &text[..offset];
    slice.char_indices().rev().next()
}

pub(crate) fn char_at_index(text: &str, mut offset: usize) -> Option<(usize, char)> {
    if offset >= text.len() {
        return None;
    }

    if !text.is_char_boundary(offset) {
        while offset > 0 && !text.is_char_boundary(offset) {
            offset -= 1;
        }
    }

    text[offset..]
        .char_indices()
        .next()
        .map(|(idx, ch)| (offset + idx, ch))
}

pub(crate) fn is_identifier_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

pub(crate) fn identifier_prefix_range(text: &str, offset: usize) -> (usize, usize) {
    let mut start = offset;
    while let Some((idx, ch)) = prev_char_index(text, start) {
        if is_identifier_char(ch) {
            start = idx;
        } else {
            break;
        }
    }

    (start, offset)
}

pub(crate) fn identifier_range_before(text: &str, mut offset: usize) -> Option<(usize, usize)> {
    if offset == 0 {
        return None;
    }

    while let Some((idx, ch)) = prev_char_index(text, offset) {
        if ch.is_whitespace() {
            offset = idx;
        } else {
            break;
        }
    }

    let end = offset;
    if end == 0 {
        return None;
    }

    let mut start = end;
    let mut cursor = end;
    while let Some((idx, ch)) = prev_char_index(text, cursor) {
        if is_identifier_char(ch) || ch == '.' {
            start = idx;
            cursor = idx;
        } else {
            break;
        }
    }

    if start == end {
        return None;
    }

    let slice = &text[start..end];
    if !slice.chars().any(is_identifier_char) {
        return None;
    }

    Some((start, end))
}

pub(crate) fn identifier_text(text: &str, range: (usize, usize)) -> String {
    text[range.0..range.1].to_string()
}

pub(crate) fn span_from_identifier(
    text: &str,
    start_offset: usize,
    name: &str,
    line_offsets: &[usize],
) -> Option<Span> {
    if name.is_empty() {
        return None;
    }

    let start_pos = offset_to_position(text, start_offset, line_offsets);
    let start_line = start_pos.line as usize + 1;
    let start_col = start_pos.character as usize + 1;
    let len_chars = name.chars().count();
    if len_chars == 0 {
        return None;
    }

    Some(Span::new(
        start_line,
        start_col,
        start_line,
        start_col + len_chars.saturating_sub(1),
    ))
}

pub(crate) fn identifier_name_at_span<'a>(
    text: &'a str,
    line_offsets: &[usize],
    span: Span,
    expected_len: usize,
) -> Option<&'a str> {
    if span.start_line == 0 || span.start_line != span.end_line || expected_len == 0 {
        return None;
    }

    let line_idx = span.start_line.checked_sub(1)?;
    if line_idx >= line_offsets.len() {
        return None;
    }

    let line_start = line_offsets[line_idx];
    let line_end = line_offsets
        .get(line_idx + 1)
        .copied()
        .unwrap_or_else(|| text.len());
    if line_start > line_end || line_end > text.len() {
        return None;
    }

    let line_text = &text[line_start..line_end];
    let start_char = span.start_col.checked_sub(1)?;
    let start_byte_in_line = nth_char_byte_index(line_text, start_char);
    let end_byte_in_line = nth_char_byte_index(line_text, start_char + expected_len);
    let start_byte = line_start.saturating_add(start_byte_in_line);
    let end_byte = line_start.saturating_add(end_byte_in_line);
    if start_byte >= end_byte || end_byte > text.len() {
        return None;
    }

    Some(&text[start_byte..end_byte])
}

pub(crate) fn simple_type_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

pub(crate) fn qualify_type_name(module_path: &str, name: &str) -> String {
    if name.contains('.') || module_path.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", module_path, name)
    }
}

pub(crate) fn named_type_name(ty: &Type) -> Option<String> {
    match &ty.kind {
        TypeKind::Named(name) => Some(name.clone()),
        TypeKind::GenericInstance { name, .. } => Some(name.clone()),
        _ => None,
    }
}

pub(crate) fn base_type_name(ty: &Type) -> Option<String> {
    match &ty.kind {
        TypeKind::Named(name) => Some(name.clone()),
        TypeKind::GenericInstance { name, .. } => Some(name.clone()),
        TypeKind::Option(_) => Some("Option".to_string()),
        TypeKind::Result(_, _) => Some("Result".to_string()),
        TypeKind::Array(_) => Some("Array".to_string()),
        TypeKind::Map(_, _) => Some("Map".to_string()),
        TypeKind::String => Some("String".to_string()),
        TypeKind::Int => Some("Int".to_string()),
        TypeKind::Float => Some("Float".to_string()),
        TypeKind::Bool => Some("Bool".to_string()),
        TypeKind::Ref(inner) | TypeKind::MutRef(inner) => base_type_name(inner),
        TypeKind::Pointer { pointee, .. } => base_type_name(pointee),
        _ => None,
    }
}

pub(crate) fn method_display_name(full: &str) -> String {
    full.rsplit(|c| c == ':' || c == '.')
        .next()
        .unwrap_or(full)
        .to_string()
}

pub(crate) fn split_type_member(name: &str) -> Option<(String, String, bool)> {
    if let Some(pos) = name.rfind(':') {
        let type_part = &name[..pos];
        let method_part = &name[pos + 1..];
        if type_part.is_empty() || method_part.is_empty() {
            return None;
        }

        return Some((type_part.to_string(), method_part.to_string(), true));
    }

    if let Some(pos) = name.rfind('.') {
        let type_part = &name[..pos];
        let method_part = &name[pos + 1..];
        if type_part.is_empty() || method_part.is_empty() {
            return None;
        }

        return Some((type_part.to_string(), method_part.to_string(), false));
    }

    None
}

pub(crate) fn span_to_range(span: Span) -> Range {
    let start_line = span.start_line.saturating_sub(1) as u32;
    let start_col = span.start_col.saturating_sub(1) as u32;
    let end_line = span.end_line.saturating_sub(1) as u32;
    let end_col = span.end_col.saturating_sub(1) as u32;
    Range {
        start: Position::new(start_line, start_col),
        end: Position::new(end_line, end_col),
    }
}

pub(crate) fn span_contains_position(span: Span, position: &Position) -> bool {
    if span.start_line == 0 {
        return false;
    }

    let line = position.line as usize + 1;
    if line < span.start_line || line > span.end_line {
        return false;
    }

    let start_col = span.start_col;
    let end_col = if span.end_col == 0 {
        span.start_col
    } else {
        span.end_col
    };
    let character = position.character as usize + 1;
    if span.start_line == span.end_line {
        character >= start_col && character <= end_col
    } else if line == span.start_line {
        character >= start_col
    } else if line == span.end_line {
        character <= end_col
    } else {
        true
    }
}

pub(crate) fn span_start_before_or_equal(span: Span, line: usize, col: usize) -> bool {
    if span.start_line == 0 {
        return false;
    }

    if span.start_line < line {
        true
    } else if span.start_line == line {
        span.start_col <= col
    } else {
        false
    }
}

pub(crate) fn span_starts_after(span: Span, line: usize, col: usize) -> bool {
    if span.start_line == 0 {
        return false;
    }

    if span.start_line > line {
        true
    } else if span.start_line == line {
        span.start_col > col
    } else {
        false
    }
}

pub(crate) fn span_overlaps_range(span: Span, range: &Range) -> bool {
    if span.start_line == 0 {
        return false;
    }

    let span_start = span.start_line.saturating_sub(1) as u32;
    let span_end = span.end_line.saturating_sub(1) as u32;
    !(span_end < range.start.line || span_start > range.end.line)
}

pub(crate) fn span_size(span: Span) -> (usize, usize) {
    let line_span = span.end_line.saturating_sub(span.start_line);
    let col_span = span.end_col.saturating_sub(span.start_col);
    (line_span, col_span)
}

pub(crate) fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '.'
}

pub(crate) fn extract_word_at_position(text: &str, position: Position) -> Option<String> {
    let line_idx = position.line as usize;
    let line = text.lines().nth(line_idx)?;
    let chars: Vec<char> = line.chars().collect();
    if chars.is_empty() {
        return None;
    }

    let mut idx = position.character as usize;
    if idx >= chars.len() {
        idx = chars.len().saturating_sub(1);
    }

    if !is_word_char(chars[idx]) {
        while idx > 0 && !is_word_char(chars[idx]) {
            idx -= 1;
        }

        if !is_word_char(chars[idx]) {
            return None;
        }
    }

    let mut start = idx;
    while start > 0 && is_word_char(chars[start - 1]) {
        start -= 1;
    }

    let mut end = idx + 1;
    while end < chars.len() && is_word_char(chars[end]) {
        end += 1;
    }

    let word: String = chars[start..end].iter().collect();
    if word.is_empty() {
        None
    } else {
        Some(word)
    }
}

pub(crate) fn span_for_identifier(
    text: &str,
    line_offsets: &[usize],
    span: Span,
    name: &str,
    approx_col: usize,
) -> Option<Span> {
    if span.start_line == 0 {
        return None;
    }

    let line_idx = span.start_line.saturating_sub(1);
    if line_idx >= line_offsets.len() {
        return None;
    }

    let line_start = line_offsets[line_idx];
    let line_end = line_offsets
        .get(line_idx + 1)
        .copied()
        .unwrap_or_else(|| text.len());
    if line_start >= line_end {
        return None;
    }

    let mut line_text = &text[line_start..line_end];
    if let Some(stripped) = line_text.strip_suffix('\n') {
        line_text = stripped;
    }

    let approx_byte = nth_char_byte_index(line_text, approx_col);
    let search_slice = if approx_byte <= line_text.len() {
        &line_text[approx_byte..]
    } else {
        ""
    };
    let byte_index = if let Some(local_offset) = search_slice.find(name) {
        approx_byte.saturating_add(local_offset)
    } else if let Some(global_offset) = line_text.find(name) {
        global_offset
    } else {
        return None;
    };
    if byte_index > line_text.len() {
        return None;
    }

    let char_start = line_text[..byte_index].chars().count();
    let len_chars = name.chars().count();
    Some(Span::new(
        span.start_line,
        char_start + 1,
        span.start_line,
        char_start + len_chars + 1,
    ))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn analyzer_lust_config() -> LustConfig {
    let mut config = LustConfig::default();
    config.enable_module("io");
    config.enable_module("os");
    config
}

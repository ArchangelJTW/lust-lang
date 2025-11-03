use crate::analysis::{AnalysisSnapshot, ModuleSnapshot};
use crate::utils::{
    compute_line_offsets, is_word_char, nth_char_byte_index, span_from_identifier, span_to_range,
};
use tower_lsp::lsp_types::{Hover, HoverContents, MarkupContent, MarkupKind, Position};

use super::completions::{
    analyze_member_method_context, format_method_signature, resolve_base_type_name_for_context,
    CompletionKind,
};

pub(crate) fn hover_for_method_call(
    snapshot: &AnalysisSnapshot,
    module: &ModuleSnapshot,
    module_path: Option<&str>,
    text: &str,
    position: Position,
    method_token: &str,
) -> Option<Hover> {
    if method_token.is_empty() {
        return None;
    }

    let line_offsets = compute_line_offsets(text);
    let line_idx = position.line as usize;
    if line_idx >= line_offsets.len() {
        return None;
    }

    let line_start = line_offsets[line_idx];
    let line_end = line_offsets
        .get(line_idx + 1)
        .copied()
        .unwrap_or_else(|| text.len());
    if line_start >= line_end || line_end > text.len() {
        return None;
    }

    let line_slice = &text[line_start..line_end];
    let chars: Vec<char> = line_slice.chars().collect();
    if chars.is_empty() {
        return None;
    }

    let mut char_idx = position.character as usize;
    if char_idx >= chars.len() {
        char_idx = chars.len().saturating_sub(1);
    }

    if !is_word_char(chars[char_idx]) {
        while char_idx > 0 && !is_word_char(chars[char_idx]) {
            char_idx -= 1;
        }

        if !is_word_char(chars[char_idx]) {
            return None;
        }
    }

    let mut start_char = char_idx;
    while start_char > 0 && is_word_char(chars[start_char - 1]) {
        start_char -= 1;
    }

    let mut end_char = char_idx + 1;
    while end_char < chars.len() && is_word_char(chars[end_char]) {
        end_char += 1;
    }

    let line_start_byte = nth_char_byte_index(line_slice, start_char);
    let line_end_byte = nth_char_byte_index(line_slice, end_char);
    if line_start_byte > line_end_byte || line_end_byte > line_slice.len() {
        return None;
    }

    let token_str = &line_slice[line_start_byte..line_end_byte];
    let desired_name = method_token.rsplit('.').next().unwrap_or(method_token);
    let method_segment = token_str.rsplit('.').next().unwrap_or(token_str);
    if method_segment != desired_name {
        return None;
    }

    let prefix_len = token_str.len().saturating_sub(method_segment.len());
    let method_start_byte = line_start_byte + prefix_len;
    let method_end_byte = method_start_byte + method_segment.len();
    let start_offset = line_start + method_start_byte;
    let end_offset = line_start + method_end_byte;
    let next_char = text[end_offset..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .next();
    if next_char != Some('(') {
        return None;
    }

    let context = analyze_member_method_context(text, end_offset)?;
    if !matches!(
        context.kind,
        CompletionKind::Method | CompletionKind::Member
    ) {
        return None;
    }

    let expect_instance = matches!(context.kind, CompletionKind::Method);
    let base_name = resolve_base_type_name_for_context(
        module,
        snapshot,
        module_path,
        text,
        &line_offsets,
        &context,
    )?;
    let methods = snapshot.methods_for_type(&base_name)?;
    let method = methods
        .iter()
        .find(|m| m.name == desired_name && m.is_instance == expect_instance)?;
    let hover_span = span_from_identifier(text, start_offset, method_segment, &line_offsets);
    let signature = format_method_signature(method);
    let mut body = format!("```lust\n{}\n```", signature);
    body.push_str(&format!("\nDefined on `{}`", method.owner));
    if !method.module_path.is_empty() {
        body.push_str(&format!("\nModule `{}`", method.module_path));
    }

    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: body,
        }),
        range: hover_span.map(span_to_range),
    })
}

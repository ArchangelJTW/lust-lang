use crate::analysis::AnalysisSnapshot;
use crate::utils::{
    base_type_name, compute_line_offsets, is_identifier_char, offset_to_position,
    position_to_offset, prev_char_index,
};
use lust::ast::FunctionParam;
use std::path::Path;
use tower_lsp::lsp_types::{
    ParameterInformation, ParameterLabel, Position, SignatureHelp, SignatureInformation,
};

pub(crate) fn signature_help(
    snapshot: &AnalysisSnapshot,
    file_path: &Path,
    position: Position,
    text: &str,
) -> Option<SignatureHelp> {
    let line_offsets = compute_line_offsets(text);
    let offset = position_to_offset(text, position, &line_offsets)?;

    let (call_info, active_param) = parse_call_context(text, offset)?;
    let module_path = snapshot.module_path_for_file(file_path);

    let (label, params, _ret_str) = match call_info {
        CallContext::Method {
            receiver,
            method_name,
            is_instance,
        } => {
            let receiver_pos = offset_to_position(text, receiver.0, &line_offsets);
            let owner_type = snapshot
                .module_for_file(file_path)
                .and_then(|m| crate::analysis::find_type_for_position(m, receiver_pos))
                .and_then(|(_, ty)| base_type_name(&ty))
                .or_else(|| {
                    let recv_name = &text[receiver.0..receiver.1];
                    snapshot
                        .struct_info_for(recv_name, module_path)
                        .map(|s| s.def.name.clone())
                })?;

            let methods = snapshot.methods_for_type(&owner_type)?;
            let method = methods
                .iter()
                .find(|m| m.name == method_name && (!is_instance || m.is_instance))?;

            let param_labels: Vec<String> = method
                .params
                .iter()
                .filter(|p| !p.is_self && p.name != "self")
                .map(format_param)
                .collect();

            let ret = method
                .return_type
                .as_ref()
                .map(|r| format!(" -> {r}"))
                .unwrap_or_default();

            let full_sig = format!("fn {}({}){}", method_name, param_labels.join(", "), ret);
            (full_sig, method.params.clone(), ret)
        }

        CallContext::Function { name } => {
            let func_info = snapshot.function_info_for(&name, module_path)?;
            let param_labels: Vec<String> = func_info
                .def
                .params
                .iter()
                .filter(|p| !p.is_self && p.name != "self")
                .map(format_param)
                .collect();

            let ret = func_info
                .def
                .return_type
                .as_ref()
                .map(|r| format!(" -> {r}"))
                .unwrap_or_default();

            let full_sig = format!("fn {}({}){}", name, param_labels.join(", "), ret);
            (full_sig, func_info.def.params.clone(), ret)
        }
    };

    let param_infos: Vec<ParameterInformation> = params
        .iter()
        .filter(|p| !p.is_self && p.name != "self")
        .map(|p| ParameterInformation {
            label: ParameterLabel::Simple(format_param(p)),
            documentation: None,
        })
        .collect();

    Some(SignatureHelp {
        signatures: vec![SignatureInformation {
            label,
            documentation: None,
            parameters: Some(param_infos),
            active_parameter: Some(active_param as u32),
        }],
        active_signature: Some(0),
        active_parameter: Some(active_param as u32),
    })
}

fn format_param(param: &FunctionParam) -> String {
    if param.is_self {
        "self".to_string()
    } else {
        format!("{}: {}", param.name, param.ty)
    }
}

enum CallContext {
    Method {
        receiver: (usize, usize),
        method_name: String,
        is_instance: bool,
    },
    Function {
        name: String,
    },
}

fn parse_call_context(text: &str, cursor_offset: usize) -> Option<(CallContext, usize)> {
    if cursor_offset == 0 {
        return None;
    }

    let mut cursor = cursor_offset;
    let mut paren_depth = 0;
    let mut comma_count = 0;
    let mut open_paren_offset = None;

    while let Some((idx, ch)) = prev_char_index(text, cursor) {
        cursor = idx;
        if ch == ')' {
            paren_depth += 1;
        } else if ch == '(' {
            if paren_depth == 0 {
                open_paren_offset = Some(idx);
                break;
            } else {
                paren_depth -= 1;
            }
        } else if ch == ',' && paren_depth == 0 {
            comma_count += 1;
        }
    }

    let open_paren = open_paren_offset?;

    // Now look backwards before open_paren to find callee
    let mut callee_end = open_paren;
    while let Some((idx, ch)) = prev_char_index(text, callee_end) {
        if ch.is_whitespace() {
            callee_end = idx;
        } else {
            break;
        }
    }

    if callee_end == 0 {
        return None;
    }

    // Extract identifier before '('
    let mut ident_start = callee_end;
    while let Some((idx, ch)) = prev_char_index(text, ident_start) {
        if is_identifier_char(ch) {
            ident_start = idx;
        } else {
            break;
        }
    }

    if ident_start == callee_end {
        return None;
    }

    let ident = &text[ident_start..callee_end];

    // Check if preceded by ':' or '.'
    if let Some((sep_idx, sep_char)) = prev_char_index(text, ident_start) {
        if sep_char == ':' || sep_char == '.' {
            // Find receiver before separator
            let mut recv_end = sep_idx;
            while let Some((idx, ch)) = prev_char_index(text, recv_end) {
                if ch.is_whitespace() {
                    recv_end = idx;
                } else {
                    break;
                }
            }
            let mut recv_start = recv_end;
            while let Some((idx, ch)) = prev_char_index(text, recv_start) {
                if is_identifier_char(ch) {
                    recv_start = idx;
                } else {
                    break;
                }
            }
            if recv_start < recv_end {
                return Some((
                    CallContext::Method {
                        receiver: (recv_start, recv_end),
                        method_name: ident.to_string(),
                        is_instance: sep_char == ':',
                    },
                    comma_count,
                ));
            }
        }
    }

    Some((
        CallContext::Function {
            name: ident.to_string(),
        },
        comma_count,
    ))
}

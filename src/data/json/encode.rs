use crate::data::json::events::JsonError;
use crate::data::{DataValue, ExactNumber};
use std::fmt::Write;

pub const DEFAULT_MAX_DEPTH: usize = 256;

pub fn to_string(value: &DataValue) -> Result<String, JsonError> {
    to_string_with_limits(value, DEFAULT_MAX_DEPTH)
}

pub fn to_string_with_limits(value: &DataValue, max_depth: usize) -> Result<String, JsonError> {
    let mut out = String::new();
    encode_value_depth(value, &mut out, 0, max_depth)?;
    Ok(out)
}

pub fn to_string_pretty(value: &DataValue) -> Result<String, JsonError> {
    to_string_pretty_with_limits(value, DEFAULT_MAX_DEPTH, 2)
}

pub fn to_string_pretty_with_limits(
    value: &DataValue,
    max_depth: usize,
    indent_step: usize,
) -> Result<String, JsonError> {
    let mut out = String::new();
    encode_value_indent_depth(value, &mut out, 0, indent_step, 0, max_depth)?;
    Ok(out)
}

fn encode_value_depth(
    value: &DataValue,
    out: &mut String,
    depth: usize,
    max_depth: usize,
) -> Result<(), JsonError> {
    if depth > max_depth {
        return Err(JsonError::MaxDepthExceeded(max_depth));
    }

    match value {
        DataValue::Null => out.push_str("null"),
        DataValue::Bool(true) => out.push_str("true"),
        DataValue::Bool(false) => out.push_str("false"),
        DataValue::Number(num) => match num {
            ExactNumber::Int(i) => write!(out, "{}", i).map_err(|e| JsonError::Io(e.to_string()))?,
            ExactNumber::UInt(u) => write!(out, "{}", u).map_err(|e| JsonError::Io(e.to_string()))?,
            ExactNumber::Float(f) => {
                if f.is_nan() || f.is_infinite() {
                    return Err(JsonError::InvalidNumber(0));
                }
                if f.fract() == 0.0 && !f.to_string().contains('e') && !f.to_string().contains('E') {
                    write!(out, "{:.1}", f).map_err(|e| JsonError::Io(e.to_string()))?;
                } else {
                    write!(out, "{}", f).map_err(|e| JsonError::Io(e.to_string()))?;
                }
            }
            ExactNumber::Decimal128(dec) => {
                write!(out, "\"{}\"", dec).map_err(|e| JsonError::Io(e.to_string()))?;
            }
        },
        DataValue::String(s) => encode_string(s, out),
        DataValue::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                encode_value_depth(item, out, depth + 1, max_depth)?;
            }
            out.push(']');
        }
        DataValue::Document(doc) => {
            out.push('{');
            for (i, (k, v)) in doc.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                encode_string(k, out);
                out.push(':');
                encode_value_depth(v, out, depth + 1, max_depth)?;
            }
            out.push('}');
        }
        DataValue::Binary(bytes) => {
            // Encode binary as byte array in standard json
            out.push('[');
            for (i, b) in bytes.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write!(out, "{}", b).map_err(|e| JsonError::Io(e.to_string()))?;
            }
            out.push(']');
        }
        DataValue::ObjectId(oid) => {
            write!(out, "{{\"$oid\":\"{}\"}}", oid.to_hex()).map_err(|e| JsonError::Io(e.to_string()))?;
        }
        DataValue::DateTimeUtc(dt) => {
            write!(out, "{{\"$date\":{}}}", dt.0).map_err(|e| JsonError::Io(e.to_string()))?;
        }
    }
    Ok(())
}

fn encode_value_indent_depth(
    value: &DataValue,
    out: &mut String,
    current_indent: usize,
    indent_step: usize,
    depth: usize,
    max_depth: usize,
) -> Result<(), JsonError> {
    if depth > max_depth {
        return Err(JsonError::MaxDepthExceeded(max_depth));
    }

    match value {
        DataValue::Array(items) if !items.is_empty() => {
            out.push_str("[\n");
            let next_indent = current_indent + indent_step;
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(",\n");
                }
                for _ in 0..next_indent {
                    out.push(' ');
                }
                encode_value_indent_depth(item, out, next_indent, indent_step, depth + 1, max_depth)?;
            }
            out.push('\n');
            for _ in 0..current_indent {
                out.push(' ');
            }
            out.push(']');
        }
        DataValue::Document(doc) if !doc.is_empty() => {
            out.push_str("{\n");
            let next_indent = current_indent + indent_step;
            for (i, (k, v)) in doc.iter().enumerate() {
                if i > 0 {
                    out.push_str(",\n");
                }
                for _ in 0..next_indent {
                    out.push(' ');
                }
                encode_string(k, out);
                out.push_str(": ");
                encode_value_indent_depth(v, out, next_indent, indent_step, depth + 1, max_depth)?;
            }
            out.push('\n');
            for _ in 0..current_indent {
                out.push(' ');
            }
            out.push('}');
        }
        _ => encode_value_depth(value, out, depth, max_depth)?,
    }
    Ok(())
}

pub fn encode_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                write!(out, "\\u{:04x}", c as u32).unwrap();
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

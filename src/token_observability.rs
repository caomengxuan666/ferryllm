//! Prompt token estimation, request-shape debugging, and cache observability helpers.

use std::fmt::Write;

use serde_json::Value;

use crate::ir::{canonical_json_string, ChatRequest, ContentBlock, Message, Tool};

pub const DEBUG_REQUEST_SHAPE_FLAG: &str = "__ferryllm_debug_request_shape";
pub const REQUEST_SHAPE_SYSTEM_WINDOW_BYTES: usize = 256;
pub const REQUEST_SHAPE_SYSTEM_WINDOW_MAX: usize = 256;

#[derive(Debug, Clone, Copy, Default)]
pub struct TokenEstimate {
    pub prompt_tokens: u64,
}

pub fn estimate_prompt_tokens(req: &ChatRequest) -> Option<TokenEstimate> {
    estimate_text_tokens(&prompt_text(req)).map(|prompt_tokens| TokenEstimate { prompt_tokens })
}

pub fn request_shape_debug_enabled(req: &ChatRequest) -> bool {
    req.extra
        .get(DEBUG_REQUEST_SHAPE_FLAG)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub fn stable_hash_hex(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    format!("{hash:016x}")
}

pub fn summarize_text(text: &str) -> String {
    format!("len={},hash={}", text.len(), stable_hash_hex(text))
}

pub fn summarize_text_windows(text: &str, window_bytes: usize, max_windows: usize) -> String {
    if text.is_empty() || window_bytes == 0 || max_windows == 0 {
        return "[]".into();
    }

    let bytes = text.as_bytes();
    let mut windows = Vec::new();
    let mut start = 0;
    while start < bytes.len() && windows.len() < max_windows {
        let mut end = (start + window_bytes).min(bytes.len());
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            end = (start + 1).min(bytes.len());
        }
        let fragment = &text[start..end];
        windows.push(format!("{}-{}:{}", start, end, stable_hash_hex(fragment)));
        start = end;
    }
    if start < bytes.len() {
        windows.push(format!("truncated_at={start}/{}", bytes.len()));
    }
    format!("[{}]", windows.join("|"))
}

pub fn summarize_text_windows_detailed(
    text: &str,
    window_bytes: usize,
    max_windows: usize,
    detail_window_bytes: usize,
    detail_max_windows: usize,
) -> String {
    if text.is_empty() || window_bytes == 0 || max_windows == 0 {
        return "[]".into();
    }

    let bytes = text.as_bytes();
    let mut windows = Vec::new();
    let mut start = 0;
    let mut window_index = 0;
    while start < bytes.len() && windows.len() < max_windows {
        let mut end = (start + window_bytes).min(bytes.len());
        while end > start && !text.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            end = (start + 1).min(bytes.len());
        }
        let fragment = &text[start..end];
        let mut entry = format!("{}-{}:{}", start, end, stable_hash_hex(fragment));
        if window_index == 0 && detail_window_bytes > 0 && detail_max_windows > 0 {
            let detail = summarize_text_windows(fragment, detail_window_bytes, detail_max_windows);
            if detail != "[]" {
                entry.push_str(&format!("{{detail={detail}}}"));
            }
        }
        windows.push(entry);
        start = end;
        window_index += 1;
    }
    if start < bytes.len() {
        windows.push(format!("truncated_at={start}/{}", bytes.len()));
    }
    format!("[{}]", windows.join("|"))
}

pub fn summarize_json(value: &Value) -> String {
    let stable = canonical_json_string(value);
    summarize_text(&stable)
}

pub fn summarize_optional_text(text: Option<&str>) -> String {
    match text {
        Some(text) if !text.is_empty() => summarize_text(text),
        Some(_) => "len=0,hash=0".into(),
        None => "-".into(),
    }
}

pub fn summarize_flag(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

pub fn push_summary_field(out: &mut String, name: &str, value: impl AsRef<str>) {
    let _ = write!(out, "{name}={},", value.as_ref());
}

fn prompt_text(req: &ChatRequest) -> String {
    let mut out = String::new();
    push_field(&mut out, "model", &req.model);
    if let Some(system) = &req.system {
        push_field(&mut out, "system", system);
    }
    for message in &req.messages {
        push_message(&mut out, message);
    }
    for tool in &req.tools {
        push_tool(&mut out, tool);
    }
    if let Some(tool_choice) = &req.tool_choice {
        push_field(&mut out, "tool_choice", &format!("{tool_choice:?}"));
    }
    out
}

fn push_message(out: &mut String, message: &Message) {
    push_field(out, "role", &format!("{:?}", message.role));
    for block in &message.content {
        match block {
            ContentBlock::Text { text, .. } => push_field(out, "text", text),
            ContentBlock::Image { media_type, .. } => push_field(out, "image", media_type),
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                push_field(out, "tool_use_id", id);
                push_field(out, "tool_use_name", name);
                push_field(out, "tool_use_input", &stable_json(input));
            }
            ContentBlock::ToolResult { id, content, .. } => {
                push_field(out, "tool_result_id", id);
                push_field(out, "tool_result", content);
            }
            ContentBlock::Thinking { thinking, .. } => push_field(out, "thinking", thinking),
            ContentBlock::RedactedThinking => push_field(out, "thinking", "redacted"),
        }
    }
}

fn push_tool(out: &mut String, tool: &Tool) {
    push_field(out, "tool_name", &tool.name);
    push_field(out, "tool_description", &tool.description);
    push_field(out, "tool_parameters", &stable_json(&tool.parameters));
}

fn push_field(out: &mut String, name: &str, value: &str) {
    out.push_str(name);
    out.push_str(": ");
    out.push_str(value);
    out.push('\n');
}

fn stable_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

#[cfg(feature = "prompt-observability")]
fn estimate_text_tokens(text: &str) -> Option<u64> {
    use tiktoken_rs::{cl100k_base, CoreBPE};

    fn count(bpe: CoreBPE, text: &str) -> u64 {
        bpe.encode_with_special_tokens(text).len() as u64
    }

    Some(count(cl100k_base().ok()?, text))
}

#[cfg(not(feature = "prompt-observability"))]
fn estimate_text_tokens(_text: &str) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detailed_windows_include_first_window_breakdown() {
        let text = "abcdefghijklmnopqrstuvwxyz0123456789";
        let summary = summarize_text_windows_detailed(text, 16, 4, 8, 4);

        assert!(summary.contains("0-16:"));
        assert!(summary.contains("{detail=[0-8:"));
        assert!(summary.contains("|8-16:"));
    }
}

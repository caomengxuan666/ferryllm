//! Entry translation: Anthropic client format ↔ unified IR.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ir::*;

// --- Deserializable request (what clients send us) ---

#[derive(Debug, Deserialize)]
pub struct AnthropicMessageRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<AnthropicSystemText>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: u32,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystemText {
    String(String),
    Array(Vec<AnthropicSystemPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicSystemPart {
    #[serde(rename = "text")]
    Text { text: String },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: Value, // string or array of blocks
        #[serde(default)]
        is_error: Option<bool>,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub ty: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub input_schema: Value,
}

// --- Serializable response (what we send back to clients) ---

#[derive(Debug, Serialize)]
pub struct AnthropicMessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub role: String,
    pub model: String,
    pub content: Vec<AnthropicRespBlock>,
    pub stop_reason: Option<String>,
    pub usage: AnthropicRespUsage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicRespBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Serialize)]
pub struct AnthropicRespUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// --- Translation: Anthropic → IR ---

pub fn anthropic_to_ir(req: &AnthropicMessageRequest) -> ChatRequest {
    let system = extract_anthropic_system(&req.system);

    let messages: Vec<Message> = req.messages.iter().map(anthropic_message_to_ir).collect();

    let tools: Vec<Tool> = req
        .tools
        .as_ref()
        .map(|ts| {
            ts.iter()
                .map(|t| Tool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                })
                .collect()
        })
        .unwrap_or_default();

    let tool_choice = req
        .tool_choice
        .as_ref()
        .and_then(parse_anthropic_tool_choice);

    ChatRequest {
        model: req.model.clone(),
        messages,
        system,
        temperature: req.temperature,
        max_tokens: Some(req.max_tokens),
        stop_sequences: req.stop_sequences.clone(),
        tools,
        tool_choice,
        stream: req.stream,
        extra: Default::default(),
    }
}

fn extract_anthropic_system(system: &Option<AnthropicSystemText>) -> Option<String> {
    match system {
        Some(AnthropicSystemText::String(s)) => Some(s.clone()),
        Some(AnthropicSystemText::Array(parts)) => {
            let text: String = parts
                .iter()
                .map(|p| match p {
                    AnthropicSystemPart::Text { text } => text.as_str(),
                })
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        None => None,
    }
}

fn anthropic_message_to_ir(msg: &AnthropicMessage) -> Message {
    let role = match msg.role.as_str() {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        _ => Role::User,
    };

    let content = anthropic_content_to_blocks(&msg.content);
    Message { role, content }
}

fn anthropic_content_to_blocks(content: &AnthropicContent) -> Vec<ContentBlock> {
    match content {
        AnthropicContent::Text(text) => {
            vec![ContentBlock::Text { text: text.clone() }]
        }
        AnthropicContent::Blocks(blocks) => blocks
            .iter()
            .map(|b| match b {
                AnthropicContentBlock::Text { text } => ContentBlock::Text { text: text.clone() },
                AnthropicContentBlock::Image { source } => ContentBlock::Image {
                    source: ImageSource::Base64 {
                        data: source.data.clone(),
                    },
                    media_type: source.media_type.clone(),
                },
                AnthropicContentBlock::ToolUse { id, name, input } => ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                },
                AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let text = match content {
                        Value::String(s) => s.clone(),
                        Value::Array(parts) => parts
                            .iter()
                            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                            .collect::<Vec<_>>()
                            .join(""),
                        _ => String::new(),
                    };
                    ContentBlock::ToolResult {
                        id: tool_use_id.clone(),
                        content: text,
                        is_error: is_error.unwrap_or(false),
                    }
                }
                AnthropicContentBlock::Thinking { thinking } => ContentBlock::Thinking {
                    thinking: thinking.clone(),
                },
            })
            .collect(),
    }
}

fn parse_anthropic_tool_choice(value: &Value) -> Option<ToolChoice> {
    match value {
        Value::String(s) => match s.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "any" => Some(ToolChoice::Any),
            _ => None,
        },
        Value::Object(obj) => {
            if let Some(ty) = obj.get("type").and_then(|t| t.as_str()) {
                match ty {
                    "auto" => Some(ToolChoice::Auto),
                    "any" => Some(ToolChoice::Any),
                    "tool" => {
                        obj.get("name")
                            .and_then(|n| n.as_str())
                            .map(|name| ToolChoice::Tool {
                                name: name.to_string(),
                            })
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

// --- Translation: IR → Anthropic response ---

pub fn ir_to_anthropic_response(ir: ChatResponse) -> AnthropicMessageResponse {
    let content: Vec<AnthropicRespBlock> = ir
        .choices
        .first()
        .and_then(|c| c.message.as_ref())
        .map(|m| {
            m.content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => {
                        Some(AnthropicRespBlock::Text { text: text.clone() })
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        Some(AnthropicRespBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        })
                    }
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    let stop_reason = ir.choices.first().and_then(|c| {
        c.finish_reason.as_ref().map(|fr| match fr {
            FinishReason::Stop => "end_turn".to_string(),
            FinishReason::Length => "max_tokens".to_string(),
            FinishReason::ToolCalls => "tool_use".to_string(),
            FinishReason::ContentFilter => "end_turn".to_string(),
        })
    });

    AnthropicMessageResponse {
        id: ir.id.clone(),
        ty: "message".into(),
        role: "assistant".into(),
        model: ir.model,
        content,
        stop_reason,
        usage: AnthropicRespUsage {
            input_tokens: ir.usage.prompt_tokens,
            output_tokens: ir.usage.completion_tokens,
        },
    }
}

// --- SSE translation ---

/// Convert a unified StreamEvent into an Anthropic SSE event string.
/// Returns (event_type, data_json) or None if the event should be skipped.
pub fn ir_to_anthropic_sse(event: StreamEvent) -> Option<(String, String)> {
    match event {
        StreamEvent::MessageStart { message_id, model } => {
            let json = serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {"input_tokens": 0, "output_tokens": 1}
                }
            });
            Some((
                "message_start".into(),
                serde_json::to_string(&json).unwrap_or_default(),
            ))
        }
        StreamEvent::ContentBlockStart {
            index,
            content_block,
        } => {
            let (block_type, mut block_json) = content_block_to_anthropic_sse(&content_block);
            block_json.insert("type".into(), Value::String(block_type.to_string()));
            let json = serde_json::json!({
                "type": "content_block_start",
                "index": index,
                "content_block": block_json
            });
            Some((
                "content_block_start".into(),
                serde_json::to_string(&json).unwrap_or_default(),
            ))
        }
        StreamEvent::ContentBlockDelta { index, delta } => {
            let (delta_type, mut delta_json) = match delta {
                ContentDelta::TextDelta { text } => ("text_delta", {
                    let mut map = serde_json::Map::new();
                    map.insert("text".into(), Value::String(text));
                    map
                }),
                ContentDelta::InputJSONDelta { partial_json } => ("input_json_delta", {
                    let mut map = serde_json::Map::new();
                    map.insert("partial_json".into(), Value::String(partial_json));
                    map
                }),
            };
            delta_json.insert("type".into(), Value::String(delta_type.to_string()));
            let json = serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": delta_json
            });
            Some((
                "content_block_delta".into(),
                serde_json::to_string(&json).unwrap_or_default(),
            ))
        }
        StreamEvent::ContentBlockStop { index } => {
            let json = serde_json::json!({
                "type": "content_block_stop",
                "index": index
            });
            Some((
                "content_block_stop".into(),
                serde_json::to_string(&json).unwrap_or_default(),
            ))
        }
        StreamEvent::MessageDelta { stop_reason, usage } => {
            let json = serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": null,
                },
                "usage": {
                    "output_tokens": usage.map(|u| u.completion_tokens).unwrap_or(0)
                }
            });
            Some((
                "message_delta".into(),
                serde_json::to_string(&json).unwrap_or_default(),
            ))
        }
        StreamEvent::MessageStop => {
            let json = serde_json::json!({"type": "message_stop"});
            Some((
                "message_stop".into(),
                serde_json::to_string(&json).unwrap_or_default(),
            ))
        }
        StreamEvent::Error { code, message } => {
            let json = serde_json::json!({
                "type": "error",
                "error": {
                    "type": code,
                    "message": message
                }
            });
            Some((
                "error".into(),
                serde_json::to_string(&json).unwrap_or_default(),
            ))
        }
    }
}

fn content_block_to_anthropic_sse(
    block: &ContentBlock,
) -> (&'static str, serde_json::Map<String, Value>) {
    let mut map = serde_json::Map::new();
    match block {
        ContentBlock::Text { text } => {
            map.insert("text".into(), Value::String(text.clone()));
            ("text", map)
        }
        ContentBlock::ToolUse { id, name, input } => {
            map.insert("id".into(), Value::String(id.clone()));
            map.insert("name".into(), Value::String(name.clone()));
            map.insert("input".into(), input.clone());
            ("tool_use", map)
        }
        _ => {
            map.insert("text".into(), Value::String(String::new()));
            ("text", map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn anthropic_tool_use_and_tool_result_preserve_ids_in_ir() {
        let raw = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_abc",
                            "name": "read_file",
                            "input": {"path": "Cargo.toml"}
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc",
                            "content": "file contents",
                            "is_error": false
                        }
                    ]
                }
            ]
        });

        let request: AnthropicMessageRequest = serde_json::from_value(raw).expect("parse request");
        let ir = anthropic_to_ir(&request);

        assert_eq!(ir.messages.len(), 2);
        assert_eq!(ir.messages[0].role, Role::Assistant);
        assert_eq!(ir.messages[1].role, Role::User);

        match &ir.messages[0].content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_abc");
                assert_eq!(name, "read_file");
                assert_eq!(input, &json!({"path": "Cargo.toml"}));
            }
            other => panic!("expected tool use, got {other:?}"),
        }

        match &ir.messages[1].content[0] {
            ContentBlock::ToolResult {
                id,
                content,
                is_error,
            } => {
                assert_eq!(id, "toolu_abc");
                assert_eq!(content, "file contents");
                assert!(!is_error);
            }
            other => panic!("expected tool result, got {other:?}"),
        }
    }
}

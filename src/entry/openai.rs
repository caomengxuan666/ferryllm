//! Entry translation: OpenAI client format ↔ unified IR.
//!
//! These types and functions handle the conversion between the OpenAI-compatible
//! JSON that clients send/receive and ferry's internal representation.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ir::*;

// --- Deserializable request (what clients send us) ---

#[derive(Debug, Deserialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<Value>, // string or array of strings
    #[serde(default)]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(default)]
    pub tool_choice: Option<Value>, // string or object
    #[serde(default)]
    pub reasoning: Option<OpenAIReasoning>,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIReasoning {
    #[serde(default)]
    pub effort: Option<ReasoningEffort>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>, // string or array of content parts
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub ty: String,
    pub function: OpenAIFunction,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIFunction {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub function: OpenAIFunctionCall,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

// --- Serializable response (what we send back to clients) ---

#[derive(Debug, Serialize)]
pub struct OpenAIChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: OpenAIUsage,
}

#[derive(Debug, Serialize)]
pub struct OpenAIChoice {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<OpenAIRespMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<OpenAIRespDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIRespMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OpenAIToolCallResp>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIRespDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<OpenAIToolCallDelta>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIToolCallResp {
    pub id: String,
    #[serde(rename = "type")]
    pub ty: String,
    pub function: OpenAIFunctionCallResp,
}

#[derive(Debug, Serialize)]
pub struct OpenAIFunctionCallResp {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct OpenAIToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub ty: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenAIFunctionCallDelta>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIFunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<OpenAIPromptTokensDetails>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIPromptTokensDetails {
    pub cached_tokens: u32,
}

// --- SSE chunk (for streaming responses to the client) ---

#[derive(Debug, Serialize)]
pub struct OpenAISseChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choices: Option<Vec<OpenAIChoice>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}

// --- Translation: OpenAI → IR ---

pub fn openai_to_ir(req: &OpenAIChatRequest) -> ChatRequest {
    let system = extract_openai_system(&req.messages);

    let messages: Vec<Message> = req
        .messages
        .iter()
        .filter(|m| m.role != "system" && m.role != "developer")
        .map(openai_message_to_ir)
        .collect();

    let tools: Vec<Tool> = req
        .tools
        .as_ref()
        .map(|ts| {
            ts.iter()
                .map(|t| Tool {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters: canonical_json(&t.function.parameters),
                    cache_control: None,
                })
                .collect()
        })
        .unwrap_or_default();

    let tool_choice = req.tool_choice.as_ref().and_then(parse_openai_tool_choice);
    let reasoning = parse_openai_reasoning(req);

    let stop_sequences = match &req.stop {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        _ => vec![],
    };

    ChatRequest {
        model: req.model.clone(),
        messages,
        system,
        system_cache_control: None,
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        stop_sequences,
        tools,
        tool_choice,
        stream: req.stream,
        prompt_cache_key: None,
        prompt_cache_retention: None,
        reasoning,
        extra: Default::default(),
    }
}

fn parse_openai_reasoning(req: &OpenAIChatRequest) -> Option<ReasoningControl> {
    let effort = req
        .reasoning
        .as_ref()
        .and_then(|reasoning| reasoning.effort.clone())
        .or_else(|| req.reasoning_effort.clone())?;
    Some(ReasoningControl {
        effort,
        budget_tokens: None,
    })
}

fn extract_openai_system(messages: &[OpenAIMessage]) -> Option<String> {
    if let Some(msg) = messages.first() {
        if msg.role == "system" || msg.role == "developer" {
            if let Some(content) = &msg.content {
                return match content {
                    Value::String(s) => Some(s.clone()),
                    Value::Array(parts) => {
                        // Look for a text part
                        parts
                            .iter()
                            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                            .collect::<Vec<_>>()
                            .join("\n")
                            .into()
                    }
                    _ => None,
                };
            }
        }
    }
    None
}

fn openai_message_to_ir(msg: &OpenAIMessage) -> Message {
    let role = match msg.role.as_str() {
        "system" | "developer" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    };

    let mut blocks = Vec::new();

    // Content blocks
    if let Some(content) = &msg.content {
        match content {
            Value::String(text) if !text.is_empty() => {
                blocks.push(ContentBlock::Text {
                    text: text.clone(),
                    cache_control: None,
                });
            }
            Value::Array(parts) => {
                for part in parts {
                    if let Some(part_type) = part.get("type").and_then(|t| t.as_str()) {
                        match part_type {
                            "text" => {
                                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                    blocks.push(ContentBlock::Text {
                                        text: text.to_string(),
                                        cache_control: None,
                                    });
                                }
                            }
                            "image_url" => {
                                if let Some(image_url) = part.get("image_url") {
                                    if let Some(url) = image_url.get("url").and_then(|u| u.as_str())
                                    {
                                        blocks.push(ContentBlock::Image {
                                            source: ImageSource::Url {
                                                url: url.to_string(),
                                            },
                                            media_type: "image/png".to_string(),
                                            cache_control: None,
                                        });
                                    }
                                }
                            }
                            "input_audio" => {
                                // Audio not yet in IR, skip
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Tool calls (assistant messages)
    if let Some(tool_calls) = &msg.tool_calls {
        for tc in tool_calls {
            let input: Value = serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);
            blocks.push(ContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input: canonical_json(&input),
                cache_control: None,
            });
        }
    }

    // Tool result (tool messages)
    if role == Role::Tool {
        if let Some(tool_call_id) = &msg.tool_call_id {
            let text = blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text, .. } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            blocks.clear();
            blocks.push(ContentBlock::ToolResult {
                id: tool_call_id.clone(),
                content: text,
                is_error: false,
                cache_control: None,
            });
        }
    }

    if blocks.is_empty() {
        blocks.push(ContentBlock::Text {
            text: String::new(),
            cache_control: None,
        });
    }

    Message {
        role,
        content: blocks,
    }
}

fn parse_openai_tool_choice(value: &Value) -> Option<ToolChoice> {
    match value {
        Value::String(s) => match s.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "required" | "any" => Some(ToolChoice::Any),
            "none" => Some(ToolChoice::None),
            _ => None,
        },
        Value::Object(obj) => {
            if let Some(func) = obj.get("function") {
                if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                    return Some(ToolChoice::Tool {
                        name: name.to_string(),
                    });
                }
            }
            None
        }
        _ => None,
    }
}

// --- Translation: IR → OpenAI response ---

pub fn ir_to_openai_response(ir: ChatResponse) -> OpenAIChatResponse {
    use std::time::{SystemTime, UNIX_EPOCH};

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let choices: Vec<OpenAIChoice> = ir
        .choices
        .into_iter()
        .map(|c| {
            let message = c.message.map(|m| {
                let (content, tool_calls) = ir_message_to_openai_resp(&m);
                OpenAIRespMessage {
                    role: ir_role_to_openai(&m.role),
                    content: Some(content),
                    tool_calls,
                }
            });

            OpenAIChoice {
                index: c.index,
                message,
                delta: None,
                finish_reason: c.finish_reason.map(|fr| finish_reason_to_openai(&fr)),
            }
        })
        .collect();

    OpenAIChatResponse {
        id: ir.id,
        object: "chat.completion".into(),
        created,
        model: ir.model,
        choices,
        usage: OpenAIUsage {
            prompt_tokens: ir.usage.prompt_tokens,
            completion_tokens: ir.usage.completion_tokens,
            total_tokens: ir.usage.total_tokens,
            prompt_tokens_details: ir
                .usage
                .cached_tokens
                .map(|cached_tokens| OpenAIPromptTokensDetails { cached_tokens }),
        },
    }
}

fn ir_message_to_openai_resp(msg: &Message) -> (String, Vec<OpenAIToolCallResp>) {
    let mut text = String::new();
    let mut tool_calls = Vec::new();

    for block in &msg.content {
        match block {
            ContentBlock::Text { text: t, .. } => text.push_str(t),
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                tool_calls.push(OpenAIToolCallResp {
                    id: id.clone(),
                    ty: "function".into(),
                    function: OpenAIFunctionCallResp {
                        name: name.clone(),
                        arguments: canonical_json_string(input),
                    },
                });
            }
            _ => {}
        }
    }

    (text, tool_calls)
}

fn ir_role_to_openai(role: &Role) -> String {
    match role {
        Role::System => "system".into(),
        Role::User => "user".into(),
        Role::Assistant => "assistant".into(),
        Role::Tool => "tool".into(),
    }
}

fn finish_reason_to_openai(fr: &FinishReason) -> String {
    match fr {
        FinishReason::Stop => "stop".into(),
        FinishReason::Length => "length".into(),
        FinishReason::ToolCalls => "tool_calls".into(),
        FinishReason::ContentFilter => "content_filter".into(),
    }
}

// --- SSE translation ---

/// Convert a unified StreamEvent into an OpenAI-style SSE `data:` line.
pub fn ir_to_openai_sse(event: StreamEvent, message_id: &str, model: &str) -> Option<String> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    match event {
        StreamEvent::MessageStart { .. } => None, // OpenAI doesn't have a message_start event
        StreamEvent::ContentBlockStart { .. } => None,
        StreamEvent::ContentBlockDelta { index, delta } => {
            let content = match delta {
                ContentDelta::TextDelta { text } => Some(text),
                ContentDelta::InputJSONDelta { .. } => None,
            };

            let chunk = OpenAISseChunk {
                id: message_id.to_string(),
                object: "chat.completion.chunk".into(),
                created,
                model: model.to_string(),
                choices: Some(vec![OpenAIChoice {
                    index,
                    message: None,
                    delta: Some(OpenAIRespDelta {
                        role: None,
                        content,
                        tool_calls: vec![],
                    }),
                    finish_reason: None,
                }]),
                usage: None,
            };
            Some(format!(
                "data: {}\n",
                serde_json::to_string(&chunk).unwrap_or_default()
            ))
        }
        StreamEvent::ContentBlockStop { .. } => None,
        StreamEvent::MessageDelta { stop_reason, usage } => {
            let chunk = OpenAISseChunk {
                id: message_id.to_string(),
                object: "chat.completion.chunk".into(),
                created,
                model: model.to_string(),
                choices: Some(vec![OpenAIChoice {
                    index: 0,
                    message: None,
                    delta: Some(OpenAIRespDelta {
                        role: None,
                        content: None,
                        tool_calls: vec![],
                    }),
                    finish_reason: stop_reason.map(|s| match s.as_str() {
                        "end_turn" => "stop".into(),
                        "max_tokens" => "length".into(),
                        "tool_use" => "tool_calls".into(),
                        "stop_sequence" => "stop".into(),
                        other => other.to_string(),
                    }),
                }]),
                usage: usage.map(|u| OpenAIUsage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                    prompt_tokens_details: u
                        .cached_tokens
                        .map(|cached_tokens| OpenAIPromptTokensDetails { cached_tokens }),
                }),
            };
            Some(format!(
                "data: {}\n",
                serde_json::to_string(&chunk).unwrap_or_default()
            ))
        }
        StreamEvent::MessageStop => Some("data: [DONE]\n".to_string()),
        StreamEvent::Error { code, message } => Some(format!(
            "data: {{\"error\": {{\"code\": \"{}\", \"message\": \"{}\"}}}}\n",
            code, message
        )),
    }
}

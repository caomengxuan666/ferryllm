//! Entry translation: OpenAI Responses API format ↔ unified IR.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ir::*;

type ToolCallCache = std::sync::Arc<std::sync::Mutex<HashMap<String, (String, Value)>>>;

#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub input: Vec<ResponsesInputItem>,
    #[serde(default)]
    pub tools: Vec<ResponsesTool>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub reasoning: Option<ResponsesReasoning>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesReasoning {
    #[serde(default)]
    pub effort: Option<String>,
}

/// An input item in the Responses API.
///
/// The Responses API supports three kinds of input items:
/// - Role-based items with `role` and `content` (user, assistant, system messages)
/// - `function_call` items with `call_id`, `name`, and `arguments` (tool calls)
/// - `function_call_output` items with `call_id` and `output` (tool results)
///
/// We implement custom deserialization to handle both formats.
#[derive(Debug)]
pub enum ResponsesInputItem {
    Message {
        role: String,
        content: Vec<ResponsesContentPart>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

impl<'de> Deserialize<'de> for ResponsesInputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;

        // Check if this is a function call item
        if let Some(item_type) = value.get("type").and_then(|v| v.as_str()) {
            if item_type == "function_call" {
                let call_id = value
                    .get("call_id")
                    .or_else(|| value.get("id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let name = value
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let arguments = match value.get("arguments") {
                    Some(Value::String(raw)) => raw.clone(),
                    Some(raw) => raw.to_string(),
                    None => "{}".into(),
                };
                return Ok(ResponsesInputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                });
            }
            if item_type == "function_call_output" {
                let call_id = value
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let output = value
                    .get("output")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                return Ok(ResponsesInputItem::FunctionCallOutput { call_id, output });
            }
        }

        // Otherwise, try to parse as a role-based message item
        let role = value
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user")
            .to_string();
        let content: Vec<ResponsesContentPart> = value
            .get("content")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        Ok(ResponsesInputItem::Message { role, content })
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "reasoning_text")]
    Reasoning { text: String },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesTool {
    Function {
        #[serde(rename = "type")]
        ty: String,
        name: String,
        description: Option<String>,
        parameters: Value,
    },
    Generic(Value),
}

// --- Serializable Responses API response ---

#[derive(Debug, Serialize, Clone)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub model: String,
    pub status: String,
    pub output: Vec<ResponsesOutputItem>,
    pub usage: ResponsesUsage,
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ResponsesOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<ResponsesOutputContent>,
        status: String,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: String,
    },
}

#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
pub enum ResponsesOutputContent {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        annotations: Vec<Value>,
    },
}

#[derive(Debug, Serialize, Clone)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

// --- Translation: OpenAI Responses → IR ---

/// Convert Responses API request to IR, with optional tool_call cache for reconstructing tool_use.
pub fn responses_to_ir(req: &ResponsesRequest) -> ChatRequest {
    responses_to_ir_with_cache(req, None)
}

pub fn responses_to_ir_with_cache(
    req: &ResponsesRequest,
    tool_call_cache: Option<&ToolCallCache>,
) -> ChatRequest {
    let system = req.instructions.clone();

    // First pass: convert all input items to IR messages
    let mut messages: Vec<Message> = Vec::new();
    for item in &req.input {
        messages.extend(responses_input_item_to_ir_with_cache(item, tool_call_cache));
    }

    // Second pass: merge placeholder messages and pair reconstructed tool calls
    // with their tool results.
    let messages = merge_tool_use_messages(messages);

    let tools = req
        .tools
        .iter()
        .map(|t| match t {
            ResponsesTool::Function {
                name,
                description,
                parameters,
                ..
            } => Tool {
                name: name.clone(),
                description: description.clone().unwrap_or_default(),
                parameters: parameters.clone(),
                cache_control: None,
            },
            ResponsesTool::Generic(raw) => {
                let name = raw
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("custom_tool")
                    .to_string();
                let description = raw
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or_default()
                    .to_string();
                // Extract the actual parameters field, or use the raw object if it looks like a schema
                let parameters = raw
                    .get("parameters")
                    .cloned()
                    .filter(|p| p.is_object())
                    .unwrap_or_else(|| {
                        // If no parameters field, check if this looks like a schema itself
                        if raw.get("type").and_then(|t| t.as_str()) == Some("object") {
                            raw.clone()
                        } else {
                            // Namespace or other non-function tool - use empty object schema
                            serde_json::json!({"type": "object", "properties": {}})
                        }
                    });
                Tool {
                    name,
                    description,
                    parameters,
                    cache_control: None,
                }
            }
        })
        .collect();

    let tool_choice = req
        .tool_choice
        .as_ref()
        .and_then(parse_responses_tool_choice);
    let reasoning = parse_responses_reasoning(&req.reasoning);

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
        max_tokens: req.max_output_tokens,
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

#[cfg(test)]
fn responses_input_item_to_ir(item: &ResponsesInputItem) -> Vec<Message> {
    responses_input_item_to_ir_with_cache(item, None)
}

fn responses_input_item_to_ir_with_cache(
    item: &ResponsesInputItem,
    tool_call_cache: Option<&ToolCallCache>,
) -> Vec<Message> {
    match item {
        ResponsesInputItem::Message { role, content } => {
            let ir_role = match role.as_str() {
                "user" | "system" => Role::User,
                "assistant" => Role::Assistant,
                _ => return vec![],
            };

            let mut blocks = Vec::new();
            for part in content {
                match part {
                    ResponsesContentPart::InputText { text }
                    | ResponsesContentPart::OutputText { text } => {
                        if !text.is_empty() {
                            blocks.push(ContentBlock::Text {
                                text: text.clone(),
                                cache_control: None,
                            });
                        }
                    }
                    ResponsesContentPart::Reasoning { .. } => {
                        // Reasoning content from previous response - skip for now
                        // (Anthropic doesn't accept thinking blocks in the input)
                    }
                }
            }

            if blocks.is_empty() {
                blocks.push(ContentBlock::Text {
                    text: String::new(),
                    cache_control: None,
                });
            }

            vec![Message {
                role: ir_role,
                content: blocks,
            }]
        }
        ResponsesInputItem::FunctionCall {
            call_id,
            name,
            arguments,
        } => {
            let input = serde_json::from_str(arguments).unwrap_or(Value::Null);
            vec![Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input: canonical_json(&input),
                    cache_control: None,
                }],
            }]
        }
        ResponsesInputItem::FunctionCallOutput { call_id, output } => {
            // Look up the tool name from cache, or use call_id as fallback
            let (tool_name, tool_input) = if let Some(cache) = tool_call_cache {
                if let Ok(cache) = cache.lock() {
                    cache
                        .get(call_id)
                        .cloned()
                        .unwrap_or_else(|| (call_id.clone(), serde_json::json!({})))
                } else {
                    (call_id.clone(), serde_json::json!({}))
                }
            } else {
                (call_id.clone(), serde_json::json!({}))
            };

            // Anthropic requires: assistant(tool_use) then user(tool_result)
            vec![
                Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: call_id.clone(),
                        name: tool_name,
                        input: tool_input,
                        cache_control: None,
                    }],
                },
                Message {
                    role: Role::Tool,
                    content: vec![ContentBlock::ToolResult {
                        id: call_id.clone(),
                        content: output.clone(),
                        is_error: false,
                        cache_control: None,
                    }],
                },
            ]
        }
    }
}

/// Merge tool_use messages reconstructed from `function_call_output` items.
///
/// Responses histories can contain placeholders around tool outputs. The first
/// pass reconstructs each `function_call_output` as `assistant(tool_use)` plus
/// `tool(tool_result)`. This pass drops empty placeholders, folds reconstructed
/// tool-use blocks into the previous assistant message, and keeps tool results
/// grouped immediately after the assistant tool calls.
fn merge_tool_use_messages(messages: Vec<Message>) -> Vec<Message> {
    let mut result: Vec<Message> = Vec::new();
    let mut pending_tool_results: Vec<ContentBlock> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::Assistant => {
                let has_tool_use = msg
                    .content
                    .iter()
                    .any(|b| matches!(b, ContentBlock::ToolUse { .. }));
                if has_tool_use {
                    if let Some(last) = result.last_mut() {
                        if last.role == Role::Assistant {
                            last.content.extend(msg.content);
                        } else {
                            result.push(msg);
                        }
                    } else {
                        result.push(msg);
                    }
                } else {
                    if !pending_tool_results.is_empty() {
                        result.push(Message {
                            role: Role::Tool,
                            content: std::mem::take(&mut pending_tool_results),
                        });
                    }

                    let has_text = msg.content.iter().any(|b| match b {
                        ContentBlock::Text { text, .. } => !text.is_empty(),
                        _ => false,
                    });
                    if has_text {
                        result.push(msg);
                    }
                }
            }
            Role::Tool => {
                for block in msg.content {
                    pending_tool_results.push(block);
                }
            }
            Role::User => {
                if !pending_tool_results.is_empty() {
                    result.push(Message {
                        role: Role::Tool,
                        content: std::mem::take(&mut pending_tool_results),
                    });
                }

                let is_empty = msg.content.iter().all(|b| match b {
                    ContentBlock::Text { text, .. } => text.is_empty(),
                    _ => false,
                });
                if !is_empty {
                    result.push(msg);
                }
            }
            _ => {
                result.push(msg);
            }
        }
    }

    // Flush remaining tool results
    if !pending_tool_results.is_empty() {
        result.push(Message {
            role: Role::Tool,
            content: pending_tool_results,
        });
    }

    result
}

fn parse_responses_tool_choice(value: &Value) -> Option<ToolChoice> {
    match value {
        Value::String(s) => match s.as_str() {
            "auto" => Some(ToolChoice::Auto),
            "required" | "any" => Some(ToolChoice::Any),
            "none" => Some(ToolChoice::None),
            _ => None,
        },
        Value::Object(obj) => {
            if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                return Some(ToolChoice::Tool {
                    name: name.to_string(),
                });
            }
            None
        }
        _ => None,
    }
}

fn parse_responses_reasoning(reasoning: &Option<ResponsesReasoning>) -> Option<ReasoningControl> {
    let effort = reasoning
        .as_ref()
        .and_then(|r| r.effort.as_ref())
        .and_then(|e| match e.as_str() {
            "none" => Some(ReasoningEffort::None),
            "minimal" => Some(ReasoningEffort::Minimal),
            "low" => Some(ReasoningEffort::Low),
            "medium" => Some(ReasoningEffort::Medium),
            "high" => Some(ReasoningEffort::High),
            "xhigh" => Some(ReasoningEffort::XHigh),
            "max" => Some(ReasoningEffort::Max),
            "ultracode" => Some(ReasoningEffort::Ultracode),
            _ => None,
        })?;
    Some(ReasoningControl {
        effort,
        budget_tokens: None,
    })
}

// --- Translation: IR → OpenAI Responses ---

pub fn ir_to_responses_response(ir: ChatResponse) -> ResponsesResponse {
    use std::time::{SystemTime, UNIX_EPOCH};

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut output = Vec::new();
    if let Some(choice) = ir.choices.first() {
        if let Some(msg) = &choice.message {
            let mut text_content = Vec::new();
            let mut tool_calls = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text, .. } if !text.is_empty() => {
                        text_content.push(ResponsesOutputContent::OutputText {
                            text: text.clone(),
                            annotations: vec![],
                        });
                    }
                    ContentBlock::ToolUse {
                        id, name, input, ..
                    } => {
                        tool_calls.push((id.clone(), name.clone(), input.clone()));
                    }
                    _ => {}
                }
            }
            if !text_content.is_empty() {
                output.push(ResponsesOutputItem::Message {
                    id: format!("msg_{}", uuid_simple()),
                    role: "assistant".into(),
                    content: text_content,
                    status: "completed".into(),
                });
            }
            for (id, name, input) in tool_calls {
                output.push(ResponsesOutputItem::FunctionCall {
                    id: id.clone(),
                    call_id: id,
                    name,
                    arguments: canonical_json_string(&input),
                    status: "completed".into(),
                });
            }
        }
    }

    ResponsesResponse {
        id: ir.id,
        object: "response".into(),
        created_at,
        model: ir.model,
        status: "completed".into(),
        output,
        usage: ResponsesUsage {
            input_tokens: ir.usage.prompt_tokens,
            output_tokens: ir.usage.completion_tokens,
            total_tokens: ir.usage.total_tokens,
        },
    }
}

fn uuid_simple() -> String {
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::AtomicU64;
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    nanos.hash(&mut h);
    seq.hash(&mut h);
    format!("{:016x}", h.finish())
}

// --- Streaming SSE conversion ---

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct ResponsesStreamState {
    pub response_id: String,
    pub message_id: String,
    pub model: String,
    pub opened_items: HashSet<u32>,
    pub accumulated_text: HashMap<u32, String>,
    /// Pending tool_use blocks: index -> (id, name, accumulated_json)
    pub pending_tool_calls: HashMap<u32, (String, String, String)>,
    /// Thinking blocks: index -> accumulated thinking text
    pub thinking_blocks: HashMap<u32, String>,
    pub pending_usage: Option<Usage>,
    pub completed_sent: Arc<AtomicBool>,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn make_sse(event_type: &str, data: serde_json::Value) -> (String, String) {
    (
        event_type.to_string(),
        serde_json::to_string(&data).unwrap_or_default(),
    )
}

/// Convert a unified StreamEvent into zero or more Responses API SSE events.
/// Returns `Vec<(event_type, data_json)>`.
pub fn ir_to_responses_sse(
    event: StreamEvent,
    state: &mut ResponsesStreamState,
) -> Vec<(String, String)> {
    match event {
        StreamEvent::MessageStart {
            message_id,
            model,
            input_tokens,
        } => {
            state.message_id = message_id;
            if !model.is_empty() {
                state.model = model;
            }
            // Store input_tokens from message_start as a fallback for usage
            if let Some(tokens) = input_tokens {
                if state.pending_usage.is_none() {
                    state.pending_usage = Some(Usage {
                        prompt_tokens: tokens,
                        completion_tokens: 0,
                        total_tokens: tokens,
                        cached_tokens: None,
                        cache_creation_input_tokens: None,
                        cache_read_input_tokens: None,
                    });
                }
            }
            vec![make_sse(
                "response.created",
                serde_json::json!({
                    "type": "response.created",
                    "response": {
                        "id": state.response_id,
                        "object": "response",
                        "created_at": now_secs(),
                        "model": state.model,
                        "status": "in_progress",
                        "output": [],
                        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                    }
                }),
            )]
        }

        StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlock::ToolUse {
                id, name, input, ..
            },
        } => {
            // Buffer the tool_use - don't emit until ContentBlockStop
            // The actual arguments come via InputJSONDelta events
            // Start with the initial input (may be empty for streaming)
            let initial_args =
                if input.is_object() && !input.as_object().is_none_or(|m| m.is_empty()) {
                    canonical_json_string(&input)
                } else {
                    String::new()
                };
            state
                .pending_tool_calls
                .insert(index, (id, name, initial_args));
            vec![]
        }

        StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlock::Thinking { .. },
        } => {
            state.opened_items.insert(index);
            state.thinking_blocks.insert(index, String::new());
            let reasoning_id = format!("rs_{}", uuid_simple());
            // Emit output_item.added FIRST to create the active reasoning item
            vec![
                make_sse(
                    "response.output_item.added",
                    serde_json::json!({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "reasoning",
                            "id": reasoning_id,
                            "summary": [],
                            "status": "in_progress"
                        }
                    }),
                ),
                make_sse(
                    "response.reasoning_summary_part.added",
                    serde_json::json!({
                        "type": "response.reasoning_summary_part.added",
                        "summary_index": 0
                    }),
                ),
            ]
        }

        StreamEvent::ContentBlockStart { .. } => vec![],

        StreamEvent::ContentBlockDelta {
            index,
            delta: ContentDelta::TextDelta { text },
        } => {
            let mut events = Vec::new();
            if !state.opened_items.contains(&index) {
                state.opened_items.insert(index);
                state.accumulated_text.insert(index, String::new());
                events.push(make_sse(
                    "response.output_item.added",
                    serde_json::json!({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": state.message_id,
                            "role": "assistant",
                            "content": [],
                            "status": "in_progress"
                        }
                    }),
                ));
                events.push(make_sse(
                    "response.content_part.added",
                    serde_json::json!({
                        "type": "response.content_part.added",
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": []
                        }
                    }),
                ));
            }
            if let Some(acc) = state.accumulated_text.get_mut(&index) {
                acc.push_str(&text);
            }
            events.push(make_sse(
                "response.output_text.delta",
                serde_json::json!({
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text
                }),
            ));
            events
        }

        StreamEvent::ContentBlockDelta {
            index,
            delta: ContentDelta::InputJSONDelta { partial_json },
        } => {
            // Accumulate partial JSON for tool_use blocks
            if let Some((_, _, ref mut args)) = state.pending_tool_calls.get_mut(&index) {
                args.push_str(&partial_json);
            }
            vec![]
        }

        StreamEvent::ContentBlockDelta {
            index,
            delta: ContentDelta::ThinkingDelta { thinking },
        } => {
            // Accumulate thinking text and emit streaming delta
            if let Some(ref mut text) = state.thinking_blocks.get_mut(&index) {
                text.push_str(&thinking);
            }
            vec![make_sse(
                "response.reasoning_summary_text.delta",
                serde_json::json!({
                    "type": "response.reasoning_summary_text.delta",
                    "summary_index": 0,
                    "delta": thinking
                }),
            )]
        }
        StreamEvent::ContentBlockStop { index } => {
            let mut events = Vec::new();

            // Check if this is a tool_use block that needs to be emitted
            if let Some((id, name, args)) = state.pending_tool_calls.remove(&index) {
                let item = serde_json::json!({
                    "type": "function_call",
                    "id": id,
                    "call_id": id,
                    "name": name,
                    "arguments": args,
                    "status": "completed"
                });
                events.push(make_sse(
                    "response.output_item.added",
                    serde_json::json!({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": item
                    }),
                ));
                events.push(make_sse(
                    "response.output_item.done",
                    serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item
                    }),
                ));
                // If this was the last pending item and message_stop was already received,
                // send response.completed now
                if state.pending_tool_calls.is_empty()
                    && state.opened_items.is_empty()
                    && state.completed_sent.load(Ordering::Relaxed)
                {
                    let usage = state.pending_usage.take().unwrap_or_default();
                    events.push(make_sse(
                        "response.completed",
                        serde_json::json!({
                            "type": "response.completed",
                            "response": {
                                "id": state.response_id,
                                "object": "response",
                                "created_at": now_secs(),
                                "model": state.model,
                                "status": "completed",
                                "output": [],
                                "usage": {
                                    "input_tokens": usage.prompt_tokens,
                                    "output_tokens": usage.completion_tokens,
                                    "total_tokens": usage.total_tokens
                                }
                            }
                        }),
                    ));
                }
                return events;
            }

            // Check if this is a thinking block
            if let Some(thinking_text) = state.thinking_blocks.remove(&index) {
                state.opened_items.remove(&index);

                // Emit reasoning output_item.done (item was already added at start)
                if !thinking_text.is_empty() {
                    let reasoning_id = format!("rs_{}", uuid_simple());
                    let item = serde_json::json!({
                        "type": "reasoning",
                        "id": reasoning_id,
                        "summary": [{"type": "summary_text", "text": thinking_text}],
                        "status": "completed"
                    });
                    events.push(make_sse(
                        "response.output_item.done",
                        serde_json::json!({
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": item
                        }),
                    ));
                }

                // If this was the last opened item and message_stop was already received,
                // send response.completed now
                if state.opened_items.is_empty() && state.completed_sent.load(Ordering::Relaxed) {
                    let usage = state.pending_usage.take().unwrap_or_default();
                    events.push(make_sse(
                        "response.completed",
                        serde_json::json!({
                            "type": "response.completed",
                            "response": {
                                "id": state.response_id,
                                "object": "response",
                                "created_at": now_secs(),
                                "model": state.model,
                                "status": "completed",
                                "output": [],
                                "usage": {
                                    "input_tokens": usage.prompt_tokens,
                                    "output_tokens": usage.completion_tokens,
                                    "total_tokens": usage.total_tokens
                                }
                            }
                        }),
                    ));
                }
                return events;
            }

            if state.opened_items.remove(&index) {
                let full_text = state.accumulated_text.remove(&index).unwrap_or_default();
                events.push(make_sse(
                    "response.output_text.done",
                    serde_json::json!({
                        "type": "response.output_text.done",
                        "output_index": 0,
                        "content_index": 0,
                        "text": full_text
                    }),
                ));
                events.push(make_sse(
                    "response.content_part.done",
                    serde_json::json!({
                        "type": "response.content_part.done",
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": full_text,
                            "annotations": []
                        }
                    }),
                ));
                events.push(make_sse(
                    "response.output_item.done",
                    serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": state.message_id,
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": full_text,
                                "annotations": []
                            }],
                            "status": "completed"
                        }
                    }),
                ));

                // If this was the last opened item and message_stop was already received,
                // send response.completed now
                if state.opened_items.is_empty() && state.completed_sent.load(Ordering::Relaxed) {
                    let usage = state.pending_usage.take().unwrap_or_default();
                    events.push(make_sse(
                        "response.completed",
                        serde_json::json!({
                            "type": "response.completed",
                            "response": {
                                "id": state.response_id,
                                "object": "response",
                                "created_at": now_secs(),
                                "model": state.model,
                                "status": "completed",
                                "output": [],
                                "usage": {
                                    "input_tokens": usage.prompt_tokens,
                                    "output_tokens": usage.completion_tokens,
                                    "total_tokens": usage.total_tokens
                                }
                            }
                        }),
                    ));
                }
            }
            events
        }

        StreamEvent::MessageDelta { usage, .. } => {
            if let Some(u) = usage {
                state.pending_usage = Some(u);
            }
            vec![]
        }

        StreamEvent::MessageStop => {
            // Only send response.completed if all opened items have received their ContentBlockStop.
            // This handles the case where upstream sends message_stop before content_block_stop.
            if !state.opened_items.is_empty() {
                // Delay sending completed until all content blocks are closed.
                // Mark message_stop_received and return empty - we'll resend when ContentBlockStop
                // detects the last item was removed.
                state.completed_sent.store(true, Ordering::Relaxed);
                return vec![];
            }
            state.completed_sent.store(true, Ordering::Relaxed);
            let usage = state.pending_usage.take().unwrap_or_default();
            vec![make_sse(
                "response.completed",
                serde_json::json!({
                    "type": "response.completed",
                    "response": {
                        "id": state.response_id,
                        "object": "response",
                        "created_at": now_secs(),
                        "model": state.model,
                        "status": "completed",
                        "output": [],
                        "usage": {
                            "input_tokens": usage.prompt_tokens,
                            "output_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens
                        }
                    }
                }),
            )]
        }

        StreamEvent::Error { code, message } => vec![make_sse(
            "error",
            serde_json::json!({
                "type": "error",
                "error": { "type": code, "message": message }
            }),
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_request_to_ir_conversion() {
        let raw = serde_json::json!({
            "model": "gpt-5.4",
            "instructions": "You are helpful.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hello"}
                    ]
                }
            ],
            "max_output_tokens": 1024,
            "stream": false
        });

        let req: ResponsesRequest = serde_json::from_value(raw).expect("parse request");
        let ir = responses_to_ir(&req);

        assert_eq!(ir.model, "gpt-5.4");
        assert_eq!(ir.system, Some("You are helpful.".into()));
        assert_eq!(ir.messages.len(), 1);
        assert_eq!(ir.max_tokens, Some(1024));
    }

    #[test]
    fn ir_to_responses_response_conversion() {
        let ir = ChatResponse {
            id: "resp_123".into(),
            model: "gpt-5.4".into(),
            choices: vec![Choice {
                index: 0,
                message: Some(Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text {
                        text: "Hello! How can I help?".into(),
                        cache_control: None,
                    }],
                }),
                delta: None,
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
                cached_tokens: None,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let response = ir_to_responses_response(ir);

        assert_eq!(response.id, "resp_123");
        assert_eq!(response.object, "response");
        assert_eq!(response.status, "completed");
        assert_eq!(response.model, "gpt-5.4");
        assert_eq!(response.output.len(), 1);
        match &response.output[0] {
            ResponsesOutputItem::Message {
                role,
                content,
                status,
                ..
            } => {
                assert_eq!(role, "assistant");
                assert_eq!(status, "completed");
                assert_eq!(content.len(), 1);
                match &content[0] {
                    ResponsesOutputContent::OutputText { text, annotations } => {
                        assert_eq!(text, "Hello! How can I help?");
                        assert!(annotations.is_empty());
                    }
                }
            }
            _ => panic!("expected message output item"),
        }
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 20);
        assert_eq!(response.usage.total_tokens, 30);
    }

    #[test]
    fn ir_to_responses_sse_text_stream() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let mut state = ResponsesStreamState {
            response_id: "resp-test".into(),
            message_id: "msg-test".into(),
            model: "gpt-5.4".into(),
            opened_items: HashSet::new(),
            accumulated_text: HashMap::new(),
            pending_tool_calls: HashMap::new(),
            thinking_blocks: HashMap::new(),
            pending_usage: None,
            completed_sent: Arc::new(AtomicBool::new(false)),
        };

        // MessageStart -> response.created
        let events = ir_to_responses_sse(
            StreamEvent::MessageStart {
                message_id: "msg-001".into(),
                model: "gpt-5.4".into(),
                input_tokens: None,
            },
            &mut state,
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "response.created");

        // First text delta -> output_item.added + content_part.added + output_text.delta
        let events = ir_to_responses_sse(
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: ContentDelta::TextDelta {
                    text: "Hello".into(),
                },
            },
            &mut state,
        );
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].0, "response.output_item.added");
        assert_eq!(events[1].0, "response.content_part.added");
        assert_eq!(events[2].0, "response.output_text.delta");

        // Subsequent text delta -> only output_text.delta
        let events = ir_to_responses_sse(
            StreamEvent::ContentBlockDelta {
                index: 0,
                delta: ContentDelta::TextDelta {
                    text: " world".into(),
                },
            },
            &mut state,
        );
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "response.output_text.delta");

        // ContentBlockStop -> output_text.done + content_part.done + output_item.done
        let events = ir_to_responses_sse(StreamEvent::ContentBlockStop { index: 0 }, &mut state);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].0, "response.output_text.done");
        assert_eq!(events[1].0, "response.content_part.done");
        assert_eq!(events[2].0, "response.output_item.done");

        // Verify accumulated text in done events
        let done_data: serde_json::Value = serde_json::from_str(&events[0].1).unwrap();
        assert_eq!(done_data["text"], "Hello world");

        // MessageDelta -> captures usage
        let events = ir_to_responses_sse(
            StreamEvent::MessageDelta {
                stop_reason: Some("end_turn".into()),
                usage: Some(Usage {
                    prompt_tokens: 100,
                    completion_tokens: 50,
                    total_tokens: 150,
                    cached_tokens: None,
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: None,
                }),
            },
            &mut state,
        );
        assert!(events.is_empty());
        assert!(state.pending_usage.is_some());

        // MessageStop -> response.completed
        let events = ir_to_responses_sse(StreamEvent::MessageStop, &mut state);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "response.completed");
        let completed_data: serde_json::Value = serde_json::from_str(&events[0].1).unwrap();
        assert_eq!(completed_data["response"]["status"], "completed");
        assert_eq!(completed_data["response"]["usage"]["input_tokens"], 100);
        assert_eq!(completed_data["response"]["usage"]["output_tokens"], 50);
        assert!(state
            .completed_sent
            .load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn function_call_output_deserialization() {
        let raw = serde_json::json!({
            "type": "function_call_output",
            "call_id": "call_abc123",
            "output": "{\"temp\": 72}"
        });

        let item: ResponsesInputItem =
            serde_json::from_value(raw).expect("parse function_call_output");
        match &item {
            ResponsesInputItem::FunctionCallOutput { call_id, output } => {
                assert_eq!(call_id, "call_abc123");
                assert_eq!(output, "{\"temp\": 72}");
            }
            _ => panic!("expected FunctionCallOutput variant"),
        }

        // Verify conversion to IR
        let ir_msgs = responses_input_item_to_ir(&item);
        assert_eq!(ir_msgs.len(), 2); // assistant(tool_use) + user(tool_result)
                                      // First message: assistant with tool_use
        assert_eq!(ir_msgs[0].role, Role::Assistant);
        // Second message: tool with tool_result
        assert_eq!(ir_msgs[1].role, Role::Tool);
        match &ir_msgs[1].content[0] {
            ContentBlock::ToolResult {
                id,
                content,
                is_error,
                ..
            } => {
                assert_eq!(id, "call_abc123");
                assert_eq!(content, "{\"temp\": 72}");
                assert!(!is_error);
            }
            _ => panic!("expected ToolResult content block"),
        }
    }

    #[test]
    fn function_call_deserialization() {
        let raw = serde_json::json!({
            "type": "function_call",
            "id": "item_abc123",
            "call_id": "call_abc123",
            "name": "Search",
            "arguments": "{\"query\":\"ferryllm\"}"
        });

        let item: ResponsesInputItem = serde_json::from_value(raw).expect("parse function_call");
        let ir_msgs = responses_input_item_to_ir(&item);
        assert_eq!(ir_msgs.len(), 1);
        assert_eq!(ir_msgs[0].role, Role::Assistant);
        match &ir_msgs[0].content[0] {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                assert_eq!(id, "call_abc123");
                assert_eq!(name, "Search");
                assert_eq!(input["query"], "ferryllm");
            }
            _ => panic!("expected ToolUse content block"),
        }
    }

    #[test]
    fn non_streaming_function_call_response_includes_call_id() {
        let ir = ChatResponse {
            id: "resp_1".into(),
            model: "gpt-5.4".into(),
            choices: vec![Choice {
                index: 0,
                message: Some(Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::ToolUse {
                        id: "call_abc123".into(),
                        name: "Search".into(),
                        input: serde_json::json!({"query":"ferryllm"}),
                        cache_control: None,
                    }],
                }),
                delta: None,
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: Usage::default(),
        };

        let response = ir_to_responses_response(ir);
        let value = serde_json::to_value(response).expect("serialize response");
        assert_eq!(value["output"][0]["type"], "function_call");
        assert_eq!(value["output"][0]["id"], "call_abc123");
        assert_eq!(value["output"][0]["call_id"], "call_abc123");
        assert_eq!(value["output"][0]["name"], "Search");
    }

    #[test]
    fn mixed_input_with_function_call_output() {
        let raw = serde_json::json!({
            "model": "gpt-4o",
            "instructions": "You are helpful.",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What's the weather?"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me check."}]
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_weather_001",
                    "output": "{\"temp\": 72, \"condition\": \"sunny\"}"
                }
            ],
            "stream": false
        });

        let req: ResponsesRequest = serde_json::from_value(raw).expect("parse request");
        let ir = responses_to_ir(&req);

        assert_eq!(ir.model, "gpt-4o");
        assert_eq!(ir.system, Some("You are helpful.".into()));
        assert_eq!(ir.messages.len(), 3);

        // First message: user
        assert_eq!(ir.messages[0].role, Role::User);
        // Second message: assistant
        assert_eq!(ir.messages[1].role, Role::Assistant);
        assert!(ir.messages[1]
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Text { text, .. } if text == "Let me check.")));
        assert!(ir.messages[1]
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { id, .. } if id == "call_weather_001")));
        // Third message: tool result from function_call_output
        assert_eq!(ir.messages[2].role, Role::Tool);
        match &ir.messages[2].content[0] {
            ContentBlock::ToolResult {
                id,
                content,
                is_error,
                ..
            } => {
                assert_eq!(id, "call_weather_001");
                assert_eq!(content, "{\"temp\": 72, \"condition\": \"sunny\"}");
                assert!(!is_error);
            }
            _ => panic!("expected ToolResult content block"),
        }
    }
}

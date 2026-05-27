//! Entry translation: OpenAI Responses API format ↔ unified IR.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ir::*;

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

#[derive(Debug, Deserialize)]
pub struct ResponsesInputItem {
    pub role: String,
    pub content: Vec<ResponsesContentPart>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
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

pub fn responses_to_ir(req: &ResponsesRequest) -> ChatRequest {
    let system = req.instructions.clone();
    let messages = req
        .input
        .iter()
        .filter_map(responses_input_item_to_ir)
        .collect();

    let tools = req
        .tools
        .iter()
        .filter_map(|t| match t {
            ResponsesTool::Function {
                name,
                description,
                parameters,
                ..
            } => Some(Tool {
                name: name.clone(),
                description: description.clone().unwrap_or_default(),
                parameters: parameters.clone(),
                cache_control: None,
            }),
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
                Some(Tool {
                    name,
                    description,
                    parameters: raw.clone(),
                    cache_control: None,
                })
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

fn responses_input_item_to_ir(item: &ResponsesInputItem) -> Option<Message> {
    let role = match item.role.as_str() {
        "user" | "system" => Role::User,
        "assistant" => Role::Assistant,
        _ => return None,
    };

    let mut blocks = Vec::new();
    for part in &item.content {
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
        }
    }

    if blocks.is_empty() {
        blocks.push(ContentBlock::Text {
            text: String::new(),
            cache_control: None,
        });
    }

    Some(Message {
        role,
        content: blocks,
    })
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
            "low" => Some(ReasoningEffort::Low),
            "medium" => Some(ReasoningEffort::Medium),
            "high" => Some(ReasoningEffort::High),
            "xhigh" => Some(ReasoningEffort::XHigh),
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
                    id,
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

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct ResponsesStreamState {
    pub response_id: String,
    pub message_id: String,
    pub model: String,
    pub opened_items: HashSet<u32>,
    pub accumulated_text: HashMap<u32, String>,
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
        StreamEvent::MessageStart { message_id, model } => {
            state.message_id = message_id;
            if !model.is_empty() {
                state.model = model;
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
            content_block: ContentBlock::ToolUse {
                id, name, input, ..
            },
            ..
        } => {
            let arguments = canonical_json_string(&input);
            let item = serde_json::json!({
                "type": "function_call",
                "id": id,
                "call_id": id,
                "name": name,
                "arguments": arguments,
                "status": "completed"
            });
            vec![
                make_sse(
                    "response.output_item.added",
                    serde_json::json!({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": item
                    }),
                ),
                make_sse(
                    "response.output_item.done",
                    serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item
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

        StreamEvent::ContentBlockDelta { .. } => vec![],

        StreamEvent::ContentBlockStop { index } => {
            if state.opened_items.remove(&index) {
                let full_text = state.accumulated_text.remove(&index).unwrap_or_default();
                vec![
                    make_sse(
                        "response.output_text.done",
                        serde_json::json!({
                            "type": "response.output_text.done",
                            "output_index": 0,
                            "content_index": 0,
                            "text": full_text
                        }),
                    ),
                    make_sse(
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
                    ),
                    make_sse(
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
                    ),
                ]
            } else {
                vec![]
            }
        }

        StreamEvent::MessageDelta { usage, .. } => {
            if let Some(u) = usage {
                state.pending_usage = Some(u);
            }
            vec![]
        }

        StreamEvent::MessageStop => {
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
            pending_usage: None,
            completed_sent: Arc::new(AtomicBool::new(false)),
        };

        // MessageStart -> response.created
        let events = ir_to_responses_sse(
            StreamEvent::MessageStart {
                message_id: "msg-001".into(),
                model: "gpt-5.4".into(),
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
}

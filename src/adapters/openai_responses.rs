use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, trace};

use crate::adapter::{Adapter, AdapterError};
use crate::ir::*;
use crate::token_observability::{
    push_summary_field, request_shape_debug_enabled, stable_hash_hex, summarize_flag,
    summarize_optional_text, summarize_text, summarize_text_windows_detailed,
    REQUEST_SHAPE_SYSTEM_WINDOW_BYTES, REQUEST_SHAPE_SYSTEM_WINDOW_MAX,
};

#[derive(Debug, Serialize)]
struct ResponsesRequest {
    model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    input: Vec<ResponsesInputItem>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ResponsesTool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponsesReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<String>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ResponsesReasoning {
    effort: String,
}

#[derive(Debug, Serialize)]
struct ResponsesInputItem {
    role: String,
    content: Vec<ResponsesContentPart>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ResponsesContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ResponsesTool {
    #[serde(rename = "function")]
    Function {
        name: String,
        description: String,
        parameters: Value,
    },
}

#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    id: String,
    model: String,
    #[serde(default)]
    output: Vec<ResponsesOutputItem>,
    #[serde(default)]
    usage: Option<ResponsesUsage>,
}

#[derive(Debug, Deserialize)]
struct ResponsesOutputItem {
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
    #[serde(default)]
    content: Vec<ResponsesOutputContent>,
}

#[derive(Debug, Deserialize)]
struct ResponsesOutputContent {
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponsesUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
    #[serde(default)]
    input_tokens_details: Option<ResponsesInputTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct ResponsesInputTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ResponsesSseEvent {
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    response: Option<ResponsesResponse>,
    #[serde(default)]
    item: Option<ResponsesOutputItem>,
    #[serde(default)]
    delta: Option<String>,
}

pub struct OpenaiResponsesAdapter {
    client: Client,
    base_url: String,
    api_key: String,
}

impl OpenaiResponsesAdapter {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
        }
    }
}

fn ir_to_responses_request(req: &ChatRequest) -> ResponsesRequest {
    let input = req
        .messages
        .iter()
        .flat_map(ir_message_to_responses_items)
        .collect();
    let tools = req
        .tools
        .iter()
        .map(|tool| ResponsesTool::Function {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: canonical_json(&tool.parameters),
        })
        .collect();

    ResponsesRequest {
        model: req.model.clone(),
        instructions: req.system.clone(),
        input,
        tools,
        tool_choice: req.tool_choice.as_ref().map(responses_tool_choice),
        reasoning: req.reasoning.as_ref().and_then(responses_reasoning_from_ir),
        temperature: req.temperature,
        max_output_tokens: req.max_tokens,
        stop: req.stop_sequences.clone(),
        prompt_cache_key: req.prompt_cache_key.clone(),
        prompt_cache_retention: req.prompt_cache_retention.clone(),
        stream: req.stream,
    }
}

fn responses_reasoning_from_ir(reasoning: &ReasoningControl) -> Option<ResponsesReasoning> {
    let effort = match reasoning.effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::XHigh => "xhigh",
    };
    Some(ResponsesReasoning {
        effort: effort.into(),
    })
}

fn responses_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => Value::String("auto".into()),
        ToolChoice::Any => Value::String("required".into()),
        ToolChoice::None => Value::String("none".into()),
        ToolChoice::Tool { name } => serde_json::json!({
            "type": "function",
            "name": name,
        }),
    }
}

fn ir_message_to_responses_items(msg: &Message) -> Vec<ResponsesInputItem> {
    let role = match msg.role {
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::User | Role::Tool => "user",
    }
    .to_string();

    let mut text_parts = Vec::new();
    for block in &msg.content {
        match block {
            ContentBlock::Text { text, .. } | ContentBlock::Thinking { thinking: text, .. } => {
                text_parts.push(text.clone());
            }
            ContentBlock::Image { source, .. } => match source {
                ImageSource::Base64 { data } => text_parts.push(format!(
                    "[image omitted: base64 payload len={}]",
                    data.len()
                )),
                ImageSource::Url { url } => text_parts.push(format!("[image omitted: {url}]")),
            },
            ContentBlock::ToolUse {
                id, name, input, ..
            } => text_parts.push(format!(
                "[tool_use id={id} name={name} input={}]",
                serde_json::to_string(input).unwrap_or_default()
            )),
            ContentBlock::ToolResult {
                id,
                content,
                is_error,
                ..
            } => text_parts.push(format!(
                "[tool_result id={id} is_error={is_error}] {content}"
            )),
            ContentBlock::RedactedThinking => {}
        }
    }

    if text_parts.is_empty() {
        text_parts.push(String::new());
    }

    vec![ResponsesInputItem {
        role,
        content: text_parts
            .into_iter()
            .map(|text| {
                if msg.role == Role::Assistant {
                    ResponsesContentPart::OutputText { text }
                } else {
                    ResponsesContentPart::InputText { text }
                }
            })
            .collect(),
    }]
}

fn responses_response_to_ir(resp: ResponsesResponse) -> ChatResponse {
    let message = Message {
        role: Role::Assistant,
        content: responses_output_to_blocks(resp.output),
    };
    ChatResponse {
        id: resp.id,
        model: resp.model,
        choices: vec![Choice {
            index: 0,
            message: Some(message),
            delta: None,
            finish_reason: Some(FinishReason::Stop),
        }],
        usage: resp.usage.map(responses_usage_to_ir).unwrap_or_default(),
    }
}

fn responses_output_to_blocks(output: Vec<ResponsesOutputItem>) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();
    for item in output {
        match item.ty.as_str() {
            "message" => {
                for content in item.content {
                    if content.ty == "output_text" {
                        if let Some(text) = content.text {
                            if !text.is_empty() {
                                blocks.push(ContentBlock::Text {
                                    text,
                                    cache_control: None,
                                });
                            }
                        }
                    }
                }
            }
            "function_call" => {
                let args = item.arguments.unwrap_or_else(|| "{}".into());
                let input = serde_json::from_str(&args).unwrap_or(Value::Null);
                blocks.push(ContentBlock::ToolUse {
                    id: item.id.unwrap_or_default(),
                    name: item.name.unwrap_or_default(),
                    input: canonical_json(&input),
                    cache_control: None,
                });
            }
            _ => {}
        }
    }

    if blocks.is_empty() {
        blocks.push(ContentBlock::Text {
            text: String::new(),
            cache_control: None,
        });
    }
    blocks
}

fn responses_usage_to_ir(usage: ResponsesUsage) -> Usage {
    let cached_tokens = usage
        .input_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens);
    Usage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        cached_tokens,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: cached_tokens,
    }
}

fn summarize_responses_request_shape(req: &ResponsesRequest) -> String {
    let serialized = serde_json::to_string(req).unwrap_or_default();
    let tools_json = serde_json::to_string(&req.tools).unwrap_or_default();
    let mut summary = String::new();
    push_summary_field(&mut summary, "model", &req.model);
    push_summary_field(&mut summary, "stream", summarize_flag(req.stream));
    push_summary_field(
        &mut summary,
        "instructions",
        summarize_optional_text(req.instructions.as_deref()),
    );
    push_summary_field(&mut summary, "input", req.input.len().to_string());
    push_summary_field(&mut summary, "tools", req.tools.len().to_string());
    push_summary_field(&mut summary, "tools_hash", stable_hash_hex(&tools_json));
    push_summary_field(
        &mut summary,
        "tool_choice",
        req.tool_choice
            .as_ref()
            .map(|value| summarize_text(&value.to_string()))
            .unwrap_or_else(|| "-".into()),
    );
    push_summary_field(
        &mut summary,
        "reasoning",
        req.reasoning
            .as_ref()
            .map(|reasoning| format!("effort={}", reasoning.effort))
            .unwrap_or_else(|| "-".into()),
    );
    push_summary_field(
        &mut summary,
        "prompt_cache_key",
        summarize_optional_text(req.prompt_cache_key.as_deref()),
    );
    push_summary_field(
        &mut summary,
        "prompt_cache_retention",
        req.prompt_cache_retention.as_deref().unwrap_or("-"),
    );
    push_summary_field(&mut summary, "body_hash", stable_hash_hex(&serialized));
    summary.push_str("input_shapes=[");
    for (index, item) in req.input.iter().enumerate() {
        if index > 0 {
            summary.push_str("; ");
        }
        summary.push_str(&summarize_responses_input_item(index, item));
    }
    summary.push(']');
    summary
}

fn summarize_responses_input_item(index: usize, item: &ResponsesInputItem) -> String {
    let mut parts = Vec::with_capacity(item.content.len());
    for part in &item.content {
        let text = match part {
            ResponsesContentPart::InputText { text } | ResponsesContentPart::OutputText { text } => {
                text
            }
        };
        if item.role == "system" {
            parts.push(format!(
                "text({};windows={})",
                summarize_text(text),
                summarize_text_windows_detailed(
                    text,
                    REQUEST_SHAPE_SYSTEM_WINDOW_BYTES,
                    REQUEST_SHAPE_SYSTEM_WINDOW_MAX,
                    64,
                    8,
                )
            ));
        } else {
            parts.push(format!("text({})", summarize_text(text)));
        }
    }
    format!(
        "#{index}:role={},content=parts({},[{}])",
        item.role,
        item.content.len(),
        parts.join("|")
    )
}

#[async_trait]
impl Adapter for OpenaiResponsesAdapter {
    fn provider_name(&self) -> &str {
        "openai_responses"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.starts_with("claude-")
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, AdapterError> {
        let native = ir_to_responses_request(request);
        let url = format!("{}/v1/responses", self.base_url);
        info!(provider = "openai_responses", model = %request.model, stream = request.stream, "sending responses request");
        trace!(provider = "openai_responses", url = %url, body_model = %native.model, "responses request prepared");
        if request_shape_debug_enabled(request) {
            debug!(
                provider = "openai_responses",
                request_shape = %summarize_responses_request_shape(&native),
                "openai responses outbound request shape"
            );
        }

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&native)
            .send()
            .await
            .map_err(|e| AdapterError::BackendError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!(provider = "openai_responses", status = %status, error = %body, "backend returned error");
            return Err(AdapterError::BackendError(format!(
                "OpenAI Responses API returned error: {}",
                body
            )));
        }

        let responses: ResponsesResponse = resp
            .json()
            .await
            .map_err(|e| AdapterError::TranslationError(e.to_string()))?;
        Ok(responses_response_to_ir(responses))
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, AdapterError>> + Send>>, AdapterError>
    {
        let mut native = ir_to_responses_request(request);
        native.stream = true;
        let url = format!("{}/v1/responses", self.base_url);
        info!(provider = "openai_responses", model = %request.model, stream = true, "sending streaming responses request");
        if request_shape_debug_enabled(request) {
            debug!(
                provider = "openai_responses",
                request_shape = %summarize_responses_request_shape(&native),
                "openai responses outbound streaming request shape"
            );
        }

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&native)
            .send()
            .await
            .map_err(|e| AdapterError::BackendError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!(provider = "openai_responses", status = %status, error = %body, "stream backend returned error");
            return Err(AdapterError::BackendError(format!(
                "OpenAI Responses API returned error: {}",
                body
            )));
        }

        let message_id = format!("resp-{}", local_id());
        let model = request.model.clone();
        let byte_stream = resp.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let _ = tx.send(Ok(StreamEvent::MessageStart {
                message_id,
                model,
            }));
            let mut buffer = String::new();
            let mut text_started = false;
            let mut tool_index = 1u32;
            let mut seen_usage: Option<Usage> = None;
            futures::pin_mut!(byte_stream);
            while let Some(chunk) = byte_stream.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(err) => {
                        let _ = tx.send(Err(AdapterError::StreamError(err.to_string())));
                        return;
                    }
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(pos) = buffer.find("\n\n") {
                    let raw = buffer[..pos].to_string();
                    buffer.drain(..pos + 2);
                    for line in raw.lines() {
                        let Some(data) = line.strip_prefix("data: ") else {
                            continue;
                        };
                        if data.trim() == "[DONE]" {
                            continue;
                        }
                        let Ok(event) = serde_json::from_str::<ResponsesSseEvent>(data) else {
                            continue;
                        };
                        handle_responses_sse_event(
                            event,
                            &tx,
                            &mut text_started,
                            &mut tool_index,
                            &mut seen_usage,
                        );
                    }
                }
            }
            if text_started {
                let _ = tx.send(Ok(StreamEvent::ContentBlockStop { index: 0 }));
            }
            let _ = tx.send(Ok(StreamEvent::MessageDelta {
                stop_reason: Some("end_turn".into()),
                usage: seen_usage,
            }));
            let _ = tx.send(Ok(StreamEvent::MessageStop));
        });

        Ok(Box::pin(UnboundedReceiverStream::new(rx)))
    }
}

fn local_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("{nanos:x}")
}

fn handle_responses_sse_event(
    event: ResponsesSseEvent,
    tx: &tokio::sync::mpsc::UnboundedSender<Result<StreamEvent, AdapterError>>,
    text_started: &mut bool,
    tool_index: &mut u32,
    seen_usage: &mut Option<Usage>,
) {
    match event.ty.as_str() {
        "response.output_text.delta" => {
            if !*text_started {
                let _ = tx.send(Ok(StreamEvent::ContentBlockStart {
                    index: 0,
                    content_block: ContentBlock::Text {
                        text: String::new(),
                        cache_control: None,
                    },
                }));
                *text_started = true;
            }
            if let Some(delta) = event.delta {
                let _ = tx.send(Ok(StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: ContentDelta::TextDelta { text: delta },
                }));
            }
        }
        "response.output_item.done" => {
            if let Some(item) = event.item {
                if item.ty == "function_call" {
                    let args = item.arguments.unwrap_or_else(|| "{}".into());
                    let input = serde_json::from_str(&args).unwrap_or(Value::Null);
                    let index = *tool_index;
                    *tool_index += 1;
                    let _ = tx.send(Ok(StreamEvent::ContentBlockStart {
                        index,
                        content_block: ContentBlock::ToolUse {
                            id: item.id.unwrap_or_default(),
                            name: item.name.unwrap_or_default(),
                            input: canonical_json(&input),
                            cache_control: None,
                        },
                    }));
                    let _ = tx.send(Ok(StreamEvent::ContentBlockStop { index }));
                }
            }
        }
        "response.completed" => {
            if let Some(response) = event.response {
                if let Some(usage) = response.usage {
                    *seen_usage = Some(responses_usage_to_ir(usage));
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sample_request() -> ChatRequest {
        ChatRequest {
            model: "gpt-5.4".into(),
            system: Some("You are concise.".into()),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "hello".into(),
                    cache_control: None,
                }],
            }],
            system_cache_control: None,
            temperature: None,
            max_tokens: Some(64),
            stop_sequences: Vec::new(),
            tools: vec![Tool {
                name: "Search".into(),
                description: "Search docs".into(),
                parameters: serde_json::json!({"type":"object"}),
                cache_control: None,
            }],
            tool_choice: None,
            stream: true,
            prompt_cache_key: Some("ferryllm".into()),
            prompt_cache_retention: None,
            reasoning: Some(ReasoningControl {
                effort: ReasoningEffort::High,
                budget_tokens: None,
            }),
            extra: HashMap::new(),
        }
    }

    #[test]
    fn responses_request_includes_reasoning_effort() {
        let native = ir_to_responses_request(&sample_request());
        let value = serde_json::to_value(&native).expect("serialize responses request");
        assert_eq!(value["reasoning"]["effort"], "high");
        assert_eq!(value["instructions"], "You are concise.");
        assert_eq!(value["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(value["tools"][0]["type"], "function");
        assert!(summarize_responses_request_shape(&native).contains("reasoning=effort=high"));
    }

    #[test]
    fn responses_usage_maps_cached_tokens() {
        let usage = responses_usage_to_ir(ResponsesUsage {
            input_tokens: 100,
            output_tokens: 10,
            total_tokens: 110,
            input_tokens_details: Some(ResponsesInputTokensDetails {
                cached_tokens: Some(64),
            }),
        });
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 10);
        assert_eq!(usage.cached_tokens, Some(64));
        assert_eq!(usage.cache_read_input_tokens, Some(64));
    }
}

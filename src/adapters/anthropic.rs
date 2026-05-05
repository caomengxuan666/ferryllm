use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adapter::{Adapter, AdapterError};
use crate::ir::*;
use tracing::{debug, error, info, trace};

/// Anthropic Messages API request body.
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<AnthropicSystem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    max_tokens: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
}

#[derive(Debug, Serialize)]
struct AnthropicSystem {
    text: String,
}

#[derive(Debug, Serialize)]
struct AnthropicThinking {
    #[serde(rename = "type")]
    ty: String,
    budget_tokens: u32,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
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
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Serialize)]
struct AnthropicImageSource {
    #[serde(rename = "type")]
    ty: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicToolChoice {
    Auto {
        #[serde(rename = "type")]
        ty: String,
    },
    Any {
        #[serde(rename = "type")]
        ty: String,
    },
    Tool {
        #[serde(rename = "type")]
        ty: String,
        name: String,
    },
    None,
}

// --- Response types ---

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct AnthropicResponse {
    id: String,
    model: String,
    #[serde(rename = "type")]
    ty: String,
    role: String,
    content: Vec<AnthropicRespBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicRespBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// --- SSE events ---

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicSseEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicMessageStart },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: AnthropicContentBlockStream,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: u32, delta: AnthropicDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMsgDelta,
        usage: Option<AnthropicMsgUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "error")]
    Error { error: AnthropicErrorBody },
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageStart {
    id: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlockStream {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(clippy::enum_variant_names)]
enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJSONDelta { partial_json: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
}

#[derive(Debug, Deserialize)]
struct AnthropicMsgDelta {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMsgUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorBody {
    #[serde(rename = "type")]
    ty: String,
    message: String,
}

pub struct AnthropicAdapter {
    client: Client,
    base_url: String,
    api_key: String,
    anthropic_version: String,
}

impl AnthropicAdapter {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
            anthropic_version: "2023-06-01".into(),
        }
    }
}

// --- Translation: IR → Anthropic ---

fn ir_to_anthropic_request(req: &ChatRequest) -> AnthropicRequest {
    let system = req
        .system
        .as_ref()
        .map(|s| AnthropicSystem { text: s.clone() });

    let messages: Vec<AnthropicMessage> = req
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(ir_message_to_anthropic)
        .collect();

    let tools: Vec<AnthropicTool> = req
        .tools
        .iter()
        .map(|t| AnthropicTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.parameters.clone(),
        })
        .collect();

    let tool_choice = req.tool_choice.as_ref().map(|tc| match tc {
        ToolChoice::Auto => AnthropicToolChoice::Auto { ty: "auto".into() },
        ToolChoice::Any => AnthropicToolChoice::Any { ty: "any".into() },
        ToolChoice::None => AnthropicToolChoice::None,
        ToolChoice::Tool { name } => AnthropicToolChoice::Tool {
            ty: "tool".into(),
            name: name.clone(),
        },
    });

    AnthropicRequest {
        model: req.model.clone(),
        messages,
        system,
        temperature: req.temperature,
        max_tokens: req.max_tokens.unwrap_or(4096),
        stop_sequences: req.stop_sequences.clone(),
        tools,
        tool_choice,
        stream: req.stream,
        thinking: None,
    }
}

fn ir_message_to_anthropic(msg: &Message) -> AnthropicMessage {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System | Role::Tool => {
            // system messages go in the top-level `system` field, not in messages.
            // Tool messages aren't used by Anthropic directly; tool_results are inline.
            "user"
        }
    };

    let content = blocks_to_anthropic(&msg.content);
    AnthropicMessage {
        role: role.into(),
        content,
    }
}

fn blocks_to_anthropic(blocks: &[ContentBlock]) -> AnthropicContent {
    // Single text block → plain string
    if blocks.len() == 1 {
        if let ContentBlock::Text { text } = &blocks[0] {
            return AnthropicContent::Text(text.clone());
        }
    }

    let anthropic_blocks: Vec<AnthropicContentBlock> = blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(AnthropicContentBlock::Text { text: text.clone() }),
            ContentBlock::Image { source, media_type } => {
                let data = match source {
                    ImageSource::Base64 { data } => data.clone(),
                    ImageSource::Url { .. } => {
                        // Anthropic doesn't support URL-based images in all versions;
                        // the caller should have resolved to base64 before reaching here.
                        return None;
                    }
                };
                Some(AnthropicContentBlock::Image {
                    source: AnthropicImageSource {
                        ty: "base64".into(),
                        media_type: media_type.clone(),
                        data,
                    },
                })
            }
            ContentBlock::ToolUse { id, name, input } => Some(AnthropicContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            }),
            ContentBlock::ToolResult {
                id,
                content,
                is_error,
            } => Some(AnthropicContentBlock::ToolResult {
                tool_use_id: id.clone(),
                content: content.clone(),
                is_error: if *is_error { Some(true) } else { None },
            }),
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking => None,
        })
        .collect();

    AnthropicContent::Blocks(anthropic_blocks)
}

// --- Translation: Anthropic → IR ---

fn anthropic_response_to_ir(resp: AnthropicResponse) -> ChatResponse {
    let content: Vec<ContentBlock> = resp.content.iter().map(anthropic_block_to_ir).collect();

    let finish_reason = match resp.stop_reason.as_deref() {
        Some("end_turn") => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("stop_sequence") => FinishReason::Stop,
        _ => FinishReason::Stop,
    };

    ChatResponse {
        id: resp.id.clone(),
        model: resp.model,
        choices: vec![Choice {
            index: 0,
            message: Some(Message {
                role: Role::Assistant,
                content,
            }),
            delta: None,
            finish_reason: Some(finish_reason),
        }],
        usage: Usage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        },
    }
}

fn anthropic_block_to_ir(block: &AnthropicRespBlock) -> ContentBlock {
    match block {
        AnthropicRespBlock::Text { text } => ContentBlock::Text { text: text.clone() },
        AnthropicRespBlock::ToolUse { id, name, input } => ContentBlock::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        },
        AnthropicRespBlock::Thinking { thinking } => ContentBlock::Thinking {
            thinking: thinking.clone(),
        },
    }
}

// --- Adapter implementation ---

#[async_trait]
impl Adapter for AnthropicAdapter {
    fn provider_name(&self) -> &str {
        "anthropic"
    }

    fn supports_model(&self, model: &str) -> bool {
        model.starts_with("claude-")
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, AdapterError> {
        let native = ir_to_anthropic_request(request);
        let url = format!("{}/v1/messages", self.base_url);
        info!(provider = "anthropic", model = %request.model, stream = request.stream, "sending chat request");
        trace!(provider = "anthropic", url = %url, body_model = %native.model, "anthropic request prepared");

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .json(&native)
            .send()
            .await
            .map_err(|e| AdapterError::BackendError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!(provider = "anthropic", status = %status, error = %body, "backend returned error");
            return Err(AdapterError::BackendError(format!(
                "Anthropic API returned error: {}",
                body
            )));
        }

        debug!(provider = "anthropic", status = %resp.status(), "backend response ok");
        let anthropic_resp: AnthropicResponse = resp
            .json()
            .await
            .map_err(|e| AdapterError::TranslationError(e.to_string()))?;

        Ok(anthropic_response_to_ir(anthropic_resp))
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, AdapterError>> + Send>>, AdapterError>
    {
        let mut native = ir_to_anthropic_request(request);
        native.stream = true;

        let url = format!("{}/v1/messages", self.base_url);
        info!(provider = "anthropic", model = %request.model, stream = true, "sending streaming request");

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .json(&native)
            .send()
            .await
            .map_err(|e| AdapterError::BackendError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!(provider = "anthropic", status = %status, error = %body, "stream backend returned error");
            return Err(AdapterError::BackendError(format!(
                "Anthropic API returned error: {}",
                body
            )));
        }

        let stream = resp.bytes_stream().map(|result| {
            let bytes = result.map_err(|e| AdapterError::StreamError(e.to_string()))?;
            let text = String::from_utf8_lossy(&bytes);
            parse_anthropic_sse_line(&text)
        });

        Ok(Box::pin(stream))
    }
}

fn parse_anthropic_sse_line(line: &str) -> Result<StreamEvent, AdapterError> {
    let line = line.trim();
    if line.is_empty() {
        return Err(AdapterError::StreamError("empty SSE line".into()));
    }

    // Anthropic SSE format: "event: <type>\ndata: <json>"
    let mut _event_type = None;
    let mut data = None;

    for sub_line in line.lines() {
        if let Some(et) = sub_line.strip_prefix("event: ") {
            _event_type = Some(et.trim().to_string());
        } else if let Some(d) = sub_line.strip_prefix("data: ") {
            data = Some(d.trim().to_string());
        }
    }

    let data = data.ok_or_else(|| AdapterError::StreamError("no data field in SSE".into()))?;
    trace!(provider = "anthropic", sse_line = %data, "parsing sse line");
    let event: AnthropicSseEvent = serde_json::from_str(&data)
        .map_err(|e| AdapterError::TranslationError(format!("parse SSE event: {e}")))?;

    match event {
        AnthropicSseEvent::MessageStart { message } => Ok(StreamEvent::MessageStart {
            message_id: message.id,
            model: String::new(),
        }),
        AnthropicSseEvent::ContentBlockStart {
            index,
            content_block,
        } => {
            let block = match content_block {
                AnthropicContentBlockStream::Text { text } => ContentBlock::Text { text },
                AnthropicContentBlockStream::ToolUse { id, name, input } => {
                    ContentBlock::ToolUse { id, name, input }
                }
                AnthropicContentBlockStream::Thinking { thinking } => {
                    ContentBlock::Thinking { thinking }
                }
            };
            Ok(StreamEvent::ContentBlockStart {
                index,
                content_block: block,
            })
        }
        AnthropicSseEvent::ContentBlockDelta { index, delta } => {
            let delta = match delta {
                AnthropicDelta::TextDelta { text } => ContentDelta::TextDelta { text },
                AnthropicDelta::InputJSONDelta { partial_json } => {
                    ContentDelta::InputJSONDelta { partial_json }
                }
                AnthropicDelta::ThinkingDelta { thinking } => {
                    ContentDelta::TextDelta { text: thinking }
                }
            };
            Ok(StreamEvent::ContentBlockDelta { index, delta })
        }
        AnthropicSseEvent::ContentBlockStop { index } => {
            Ok(StreamEvent::ContentBlockStop { index })
        }
        AnthropicSseEvent::MessageDelta { delta, usage } => Ok(StreamEvent::MessageDelta {
            stop_reason: delta.stop_reason,
            usage: usage.map(|u| Usage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.input_tokens + u.output_tokens,
            }),
        }),
        AnthropicSseEvent::MessageStop => Ok(StreamEvent::MessageStop),
        AnthropicSseEvent::Error { error } => Ok(StreamEvent::Error {
            code: error.ty,
            message: error.message,
        }),
    }
}

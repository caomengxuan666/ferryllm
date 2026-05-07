use std::collections::HashMap;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, trace};

use crate::adapter::{Adapter, AdapterError};
use crate::ir::*;
use crate::token_observability::{
    request_shape_debug_enabled, stable_hash_hex, summarize_flag, summarize_optional_text,
    summarize_text,
};

#[derive(Debug, Serialize)]
struct GeminiRequest {
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tools: Vec<GeminiTool>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    tool_config: Option<GeminiToolConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(rename = "inlineData", skip_serializing_if = "Option::is_none")]
    inline_data: Option<GeminiInlineData>,
    #[serde(rename = "functionCall", skip_serializing_if = "Option::is_none")]
    function_call: Option<GeminiFunctionCall>,
    #[serde(rename = "functionResponse", skip_serializing_if = "Option::is_none")]
    function_response: Option<GeminiFunctionResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiInlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    args: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: Value,
}

#[derive(Debug, Serialize)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(rename = "maxOutputTokens", skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(rename = "stopSequences", default, skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiThinkingConfig {
    #[serde(rename = "thinkingBudget", skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
    #[serde(rename = "thinkingLevel", skip_serializing_if = "Option::is_none")]
    thinking_level: Option<String>,
}

#[derive(Debug, Serialize)]
struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
struct GeminiToolConfig {
    #[serde(rename = "functionCallingConfig")]
    function_calling_config: GeminiFunctionCallingConfig,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionCallingConfig {
    mode: String,
    #[serde(
        rename = "allowedFunctionNames",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    allowed_function_names: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata", default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    #[serde(default)]
    content: Option<GeminiContentResponse>,
    #[serde(rename = "finishReason", default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiContentResponse {
    #[serde(default)]
    #[allow(dead_code)]
    role: Option<String>,
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u32,
    #[serde(rename = "totalTokenCount", default)]
    total_token_count: u32,
    #[serde(rename = "cachedContentTokenCount", default)]
    cached_content_token_count: Option<u32>,
}

pub struct GeminiAdapter {
    client: Client,
    base_url: String,
    api_key: String,
}

impl GeminiAdapter {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
        }
    }
}

fn ir_to_gemini_request(req: &ChatRequest) -> GeminiRequest {
    let mut tool_names_by_id = HashMap::new();

    let system_instruction = req.system.as_ref().map(|system| GeminiContent {
        role: None,
        parts: vec![GeminiPart {
            text: Some(system.clone()),
            inline_data: None,
            function_call: None,
            function_response: None,
        }],
    });

    let contents = req
        .messages
        .iter()
        .filter_map(|message| ir_message_to_gemini_content(message, &mut tool_names_by_id))
        .collect();

    let tools = if req.tools.is_empty() {
        Vec::new()
    } else {
        vec![GeminiTool {
            function_declarations: req
                .tools
                .iter()
                .map(|tool| GeminiFunctionDeclaration {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: canonical_json(&tool.parameters),
                })
                .collect(),
        }]
    };

    let tool_config = req.tool_choice.as_ref().map(|choice| GeminiToolConfig {
        function_calling_config: match choice {
            ToolChoice::Auto => GeminiFunctionCallingConfig {
                mode: "AUTO".into(),
                allowed_function_names: Vec::new(),
            },
            ToolChoice::Any => GeminiFunctionCallingConfig {
                mode: "ANY".into(),
                allowed_function_names: Vec::new(),
            },
            ToolChoice::None => GeminiFunctionCallingConfig {
                mode: "NONE".into(),
                allowed_function_names: Vec::new(),
            },
            ToolChoice::Tool { name } => GeminiFunctionCallingConfig {
                mode: "ANY".into(),
                allowed_function_names: vec![name.clone()],
            },
        },
    });

    let generation_config = if req.temperature.is_some()
        || req.max_tokens.is_some()
        || !req.stop_sequences.is_empty()
        || req.reasoning.is_some()
    {
        Some(GeminiGenerationConfig {
            temperature: req.temperature,
            max_output_tokens: req.max_tokens,
            stop_sequences: req.stop_sequences.clone(),
            thinking_config: req
                .reasoning
                .as_ref()
                .and_then(gemini_thinking_from_ir),
        })
    } else {
        None
    };

    GeminiRequest {
        system_instruction,
        contents,
        generation_config,
        tools,
        tool_config,
    }
}

fn gemini_thinking_from_ir(reasoning: &ReasoningControl) -> Option<GeminiThinkingConfig> {
    if reasoning.effort == ReasoningEffort::None {
        return Some(GeminiThinkingConfig {
            thinking_budget: Some(0),
            thinking_level: None,
        });
    }

    if let Some(budget) = reasoning.budget_tokens {
        return Some(GeminiThinkingConfig {
            thinking_budget: Some(budget),
            thinking_level: None,
        });
    }

    let thinking_level = match reasoning.effort {
        ReasoningEffort::Low | ReasoningEffort::Medium => "low",
        ReasoningEffort::High | ReasoningEffort::XHigh => "high",
        ReasoningEffort::None => "low",
    };
    Some(GeminiThinkingConfig {
        thinking_budget: None,
        thinking_level: Some(thinking_level.into()),
    })
}

fn ir_message_to_gemini_content(
    message: &Message,
    tool_names_by_id: &mut HashMap<String, String>,
) -> Option<GeminiContent> {
    let role = match message.role {
        Role::User | Role::Tool => "user",
        Role::Assistant => "model",
        Role::System => return None,
    }
    .to_string();

    let mut parts = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::Text { text, .. } | ContentBlock::Thinking { thinking: text, .. } => {
                parts.push(GeminiPart {
                    text: Some(text.clone()),
                    inline_data: None,
                    function_call: None,
                    function_response: None,
                });
            }
            ContentBlock::Image { source, media_type, .. } => match source {
                ImageSource::Base64 { data } => parts.push(GeminiPart {
                    text: None,
                    inline_data: Some(GeminiInlineData {
                        mime_type: media_type.clone(),
                        data: data.clone(),
                    }),
                    function_call: None,
                    function_response: None,
                }),
                ImageSource::Url { url } => parts.push(GeminiPart {
                    text: Some(format!("[image omitted: {url}]")),
                    inline_data: None,
                    function_call: None,
                    function_response: None,
                }),
            },
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                tool_names_by_id.insert(id.clone(), name.clone());
                parts.push(GeminiPart {
                    text: None,
                    inline_data: None,
                    function_call: Some(GeminiFunctionCall {
                        name: name.clone(),
                        args: Some(canonical_json(input)),
                    }),
                    function_response: None,
                });
            }
            ContentBlock::ToolResult {
                id,
                content,
                is_error,
                ..
            } => {
                let name = tool_names_by_id
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| id.clone());
                let response = if *is_error {
                    json!({ "error": content })
                } else {
                    json!({ "text": content })
                };
                parts.push(GeminiPart {
                    text: None,
                    inline_data: None,
                    function_call: None,
                    function_response: Some(GeminiFunctionResponse { name, response }),
                });
            }
            ContentBlock::RedactedThinking => {}
        }
    }

    if parts.is_empty() {
        parts.push(GeminiPart {
            text: Some(String::new()),
            inline_data: None,
            function_call: None,
            function_response: None,
        });
    }

    Some(GeminiContent {
        role: Some(role),
        parts,
    })
}

fn gemini_response_to_ir(resp: GeminiResponse) -> ChatResponse {
    let mut usage = Usage::default();
    if let Some(meta) = resp.usage_metadata {
        usage.prompt_tokens = meta.prompt_token_count;
        usage.completion_tokens = meta.candidates_token_count;
        usage.total_tokens = meta.total_token_count;
        usage.cached_tokens = meta.cached_content_token_count;
    }

    let candidate = resp.candidates.into_iter().next();
    let (blocks, finish_reason) = if let Some(candidate) = candidate {
        let blocks = candidate
            .content
            .map(|content| gemini_content_to_blocks(content.parts))
            .unwrap_or_default();
        let finish_reason = candidate.finish_reason.as_deref().and_then(gemini_finish_reason);
        (blocks, finish_reason)
    } else {
        (Vec::new(), None)
    };

    let message = Message {
        role: Role::Assistant,
        content: if blocks.is_empty() {
            vec![ContentBlock::Text {
                text: String::new(),
                cache_control: None,
            }]
        } else {
            blocks
        },
    };

    ChatResponse {
        id: format!("gemini-{}", local_id()),
        model: String::new(),
        choices: vec![Choice {
            index: 0,
            message: Some(message),
            delta: None,
            finish_reason,
        }],
        usage,
    }
}

fn gemini_content_to_blocks(parts: Vec<GeminiPart>) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();
    for (index, part) in parts.into_iter().enumerate() {
        if let Some(text) = part.text {
            if !text.is_empty() {
                blocks.push(ContentBlock::Text {
                    text,
                    cache_control: None,
                });
            }
        }
        if let Some(function_call) = part.function_call {
            blocks.push(ContentBlock::ToolUse {
                id: format!("gemini-call-{index}"),
                name: function_call.name,
                input: function_call.args.unwrap_or(Value::Null),
                cache_control: None,
            });
        }
    }
    blocks
}

fn gemini_finish_reason(reason: &str) -> Option<FinishReason> {
    match reason {
        "STOP" => Some(FinishReason::Stop),
        "MAX_TOKENS" => Some(FinishReason::Length),
        "MALFORMED_FUNCTION_CALL" => Some(FinishReason::ToolCalls),
        "SAFETY" | "RECITATION" | "LANGUAGE" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
            Some(FinishReason::ContentFilter)
        }
        _ => Some(FinishReason::Stop),
    }
}

fn gemini_request_url(base_url: &str, model: &str, stream: bool) -> String {
    let base = base_url.trim_end_matches('/');
    let model = model.strip_prefix("models/").unwrap_or(model);
    if stream {
        format!("{base}/v1beta/models/{model}:streamGenerateContent?alt=sse")
    } else {
        format!("{base}/v1beta/models/{model}:generateContent")
    }
}

fn summarize_gemini_request_shape(req: &GeminiRequest) -> String {
    let serialized = serde_json::to_string(req).unwrap_or_default();
    format!(
        "model={};system={};contents={};tools={};tool_config={};payload_hash={}",
        req.contents
            .first()
            .and_then(|content| content.role.as_deref())
            .unwrap_or("-"),
        summarize_optional_text(
            req.system_instruction
                .as_ref()
                .and_then(|content| content.parts.first())
                .and_then(|part| part.text.as_deref())
        ),
        req.contents.len(),
        req.tools.len(),
        summarize_flag(req.tool_config.is_some()),
        stable_hash_hex(&serialized)
    )
}

fn summarize_gemini_request(req: &GeminiRequest) -> String {
    let mut summary = String::new();
    if let Some(system) = &req.system_instruction {
        let text = system
            .parts
            .first()
            .and_then(|part| part.text.as_deref())
            .unwrap_or("");
        summary.push_str(&format!(
            "system={} ",
            summarize_text(text)
        ));
    }
    summary.push_str(&format!(
        "contents={} tools={} payload_hash={}",
        req.contents.len(),
        req.tools.len(),
        stable_hash_hex(&serde_json::to_string(req).unwrap_or_default())
    ));
    summary
}

fn parse_gemini_sse_frame(frame: &str) -> Result<Option<GeminiResponse>, AdapterError> {
    let mut data_lines = Vec::new();
    for line in frame.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data:") {
            let payload = data.trim();
            if payload == "[DONE]" {
                return Ok(None);
            }
            data_lines.push(payload.to_string());
        }
    }

    if data_lines.is_empty() {
        return Err(AdapterError::StreamError("missing data in Gemini SSE frame".into()));
    }

    let payload = data_lines.join("\n");
    let response = serde_json::from_str(&payload)
        .map_err(|e| AdapterError::TranslationError(format!("failed to parse Gemini SSE frame: {e}")))?;
    Ok(Some(response))
}

fn local_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:x}{:x}", now.as_secs(), now.subsec_nanos())
}

#[async_trait]
impl Adapter for GeminiAdapter {
    fn provider_name(&self) -> &str {
        "gemini"
    }

    fn supports_model(&self, model: &str) -> bool {
        model.starts_with("gemini-") || model.starts_with("models/gemini-")
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, AdapterError> {
        let native = ir_to_gemini_request(request);
        let url = gemini_request_url(&self.base_url, &request.model, false);
        info!(provider = "gemini", model = %request.model, stream = request.stream, "sending chat request");
        trace!(provider = "gemini", url = %url, "gemini request prepared");
        if request_shape_debug_enabled(request) {
            debug!(
                provider = "gemini",
                request_shape = %summarize_gemini_request_shape(&native),
                "gemini outbound request shape"
            );
        }
        trace!(provider = "gemini", request = %summarize_gemini_request(&native), "gemini outbound request");

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&native)
            .send()
            .await
            .map_err(|e| AdapterError::BackendError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!(provider = "gemini", status = %status, error = %body, "backend returned error");
            return Err(AdapterError::BackendError(format!(
                "Gemini API returned error: {}",
                body
            )));
        }

        let gemini_resp: GeminiResponse = resp
            .json()
            .await
            .map_err(|e| AdapterError::TranslationError(e.to_string()))?;
        Ok(gemini_response_to_ir(gemini_resp))
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, AdapterError>> + Send>>, AdapterError>
    {
        let native = ir_to_gemini_request(request);
        let url = gemini_request_url(&self.base_url, &request.model, true);
        info!(provider = "gemini", model = %request.model, stream = true, "sending streaming request");
        if request_shape_debug_enabled(request) {
            debug!(
                provider = "gemini",
                request_shape = %summarize_gemini_request_shape(&native),
                "gemini outbound streaming request shape"
            );
        }
        trace!(provider = "gemini", request = %summarize_gemini_request(&native), "gemini outbound streaming request");

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&native)
            .send()
            .await
            .map_err(|e| AdapterError::BackendError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!(provider = "gemini", status = %status, error = %body, "stream backend returned error");
            return Err(AdapterError::BackendError(format!(
                "Gemini API returned error: {}",
                body
            )));
        }

        let message_id = format!("msg_{}", local_id());
        let model = request.model.clone();
        let byte_stream = resp.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let _ = tx.send(Ok(StreamEvent::MessageStart { message_id, model }));
            let mut buffer = String::new();
            let mut text_started = false;
            let mut text_index = None::<u32>;
            let mut next_output_index: u32 = 0;
            let mut last_usage: Option<Usage> = None;
            let mut stop_reason: Option<String> = None;

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
                    let frame = buffer[..pos].to_string();
                    buffer.drain(..pos + 2);
                    let response = match parse_gemini_sse_frame(&frame) {
                        Ok(Some(response)) => response,
                        Ok(None) => continue,
                        Err(err) => {
                            let _ = tx.send(Err(err));
                            return;
                        }
                    };

                    if response.candidates.is_empty() && response.usage_metadata.is_none() {
                        continue;
                    }

                    if let Some(meta) = response.usage_metadata {
                        last_usage = Some(Usage {
                            prompt_tokens: meta.prompt_token_count,
                            completion_tokens: meta.candidates_token_count,
                            total_tokens: meta.total_token_count,
                            cached_tokens: meta.cached_content_token_count,
                            cache_creation_input_tokens: None,
                            cache_read_input_tokens: meta.cached_content_token_count,
                        });
                    }

                    for candidate in response.candidates {
                        if let Some(reason) = candidate.finish_reason {
                            stop_reason = Some(match gemini_finish_reason(&reason) {
                                Some(FinishReason::Stop) => "stop".into(),
                                Some(FinishReason::Length) => "length".into(),
                                Some(FinishReason::ToolCalls) => "tool_calls".into(),
                                Some(FinishReason::ContentFilter) => "content_filter".into(),
                                None => reason.to_lowercase(),
                            });
                        }

                        let Some(content) = candidate.content else {
                            continue;
                        };
                        for part in content.parts {
                            if let Some(text) = part.text {
                                if text.is_empty() {
                                    continue;
                                }
                                let index = text_index.unwrap_or_else(|| {
                                    let next = next_output_index;
                                    next_output_index += 1;
                                    text_index = Some(next);
                                    let _ = tx.send(Ok(StreamEvent::ContentBlockStart {
                                        index: next,
                                        content_block: ContentBlock::Text {
                                            text: String::new(),
                                            cache_control: None,
                                        },
                                    }));
                                    text_started = true;
                                    next
                                });
                                let _ = tx.send(Ok(StreamEvent::ContentBlockDelta {
                                    index,
                                    delta: ContentDelta::TextDelta { text },
                                }));
                            }

                            if let Some(function_call) = part.function_call {
                                if text_started {
                                    if let Some(index) = text_index.take() {
                                        let _ = tx.send(Ok(StreamEvent::ContentBlockStop { index }));
                                    }
                                    text_started = false;
                                }
                                let index = next_output_index;
                                next_output_index += 1;
                                let _ = tx.send(Ok(StreamEvent::ContentBlockStart {
                                    index,
                                    content_block: ContentBlock::ToolUse {
                                        id: format!("gemini-call-{index}"),
                                        name: function_call.name,
                                        input: function_call.args.unwrap_or(Value::Null),
                                        cache_control: None,
                                    },
                                }));
                                let _ = tx.send(Ok(StreamEvent::ContentBlockStop { index }));
                            }
                        }
                    }
                }
            }

            if text_started {
                if let Some(index) = text_index {
                    let _ = tx.send(Ok(StreamEvent::ContentBlockStop { index }));
                }
            }
            let _ = tx.send(Ok(StreamEvent::MessageDelta {
                stop_reason,
                usage: last_usage,
            }));
            let _ = tx.send(Ok(StreamEvent::MessageStop));
        });

        Ok(Box::pin(UnboundedReceiverStream::new(rx)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn gemini_request_preserves_tools_and_system_instruction() {
        let req = ChatRequest {
            model: "gemini-2.5-flash".into(),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "hello".into(),
                    cache_control: None,
                }],
            }],
            system: Some("be brief".into()),
            system_cache_control: None,
            temperature: Some(0.2),
            max_tokens: Some(32),
            stop_sequences: vec!["STOP".into()],
            tools: vec![Tool {
                name: "lookup".into(),
                description: "look up".into(),
                parameters: json!({"type":"object","properties":{"q":{"type":"string"}}}),
                cache_control: None,
            }],
            tool_choice: Some(ToolChoice::Any),
            stream: false,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: Some(ReasoningControl {
                effort: ReasoningEffort::Medium,
                budget_tokens: Some(2048),
            }),
            extra: Default::default(),
        };

        let native = ir_to_gemini_request(&req);
        assert_eq!(
            native
                .system_instruction
                .as_ref()
                .and_then(|content| content.parts.first())
                .and_then(|part| part.text.as_deref()),
            Some("be brief")
        );
        assert_eq!(native.contents.len(), 1);
        assert_eq!(native.tools.len(), 1);
        assert_eq!(
            native.tool_config.as_ref().map(|config| config.function_calling_config.mode.as_str()),
            Some("ANY")
        );
    }

    #[test]
    fn gemini_response_maps_text_and_usage() {
        let resp = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: Some(GeminiContentResponse {
                    role: Some("model".into()),
                    parts: vec![GeminiPart {
                        text: Some("pong".into()),
                        inline_data: None,
                        function_call: None,
                        function_response: None,
                    }],
                }),
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: 10,
                candidates_token_count: 4,
                total_token_count: 14,
                cached_content_token_count: Some(3),
            }),
        };

        let ir = gemini_response_to_ir(resp);
        assert!(matches!(ir.choices[0].finish_reason, Some(FinishReason::Stop)));
        assert_eq!(ir.usage.prompt_tokens, 10);
        assert_eq!(ir.usage.cached_tokens, Some(3));
        match ir.choices[0].message.as_ref().unwrap().content[0] {
            ContentBlock::Text { ref text, .. } => assert_eq!(text, "pong"),
            _ => panic!("expected text"),
        }
    }
}

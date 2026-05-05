use std::collections::{hash_map::Entry, HashMap, VecDeque};
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adapter::{Adapter, AdapterError};
use crate::ir::*;
use tracing::{debug, error, info, trace, warn};

/// OpenAI-native request body (matches their `/v1/chat/completions` schema).
#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAITool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "should_skip_content")]
    content: OpenAIContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAIContent {
    Text(String),
    MultiPart(Vec<OpenAIContentPart>),
}

fn should_skip_content(content: &OpenAIContent) -> bool {
    matches!(content, OpenAIContent::Text(text) if text.is_empty())
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum OpenAIContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: OpenAIImageUrl },
}

#[derive(Debug, Serialize)]
struct OpenAIImageUrl {
    url: String,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    ty: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    ty: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAIToolChoice {
    Str(String),
    Tool {
        #[serde(rename = "type")]
        ty: String,
        function: OpenAIToolChoiceFunction,
    },
}

#[derive(Debug, Serialize)]
struct OpenAIToolChoiceFunction {
    name: String,
}

// --- Response types ---

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    id: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    index: u32,
    message: Option<OpenAIRespMessage>,
    delta: Option<OpenAIRespDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIRespMessage {
    role: Option<String>,
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAIToolCallResp>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIRespDelta {
    role: Option<String>,
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAIToolCallDelta>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIToolCallResp {
    id: String,
    #[serde(rename = "type")]
    ty: String,
    function: OpenAIFunctionCallResp,
}

#[derive(Debug, Deserialize)]
struct OpenAIFunctionCallResp {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAIToolCallDelta {
    index: u32,
    id: Option<String>,
    #[serde(rename = "type")]
    ty: Option<String>,
    function: Option<OpenAIFunctionCallDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAIFunctionCallDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// --- SSE chunk ---

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpenAISseChunk {
    id: Option<String>,
    model: Option<String>,
    object: Option<String>,
    choices: Option<Vec<OpenAIChoice>>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Default)]
struct OpenAIToolStreamState {
    started: HashMap<u32, String>,
}

pub struct OpenaiAdapter {
    client: Client,
    base_url: String,
    api_key: String,
}

impl OpenaiAdapter {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
        }
    }
}

// --- Translation functions ---

fn ir_to_openai_request(req: &ChatRequest) -> OpenAIChatRequest {
    let system_msg = req.system.as_ref().map(|s| OpenAIMessage {
        role: "system".into(),
        content: OpenAIContent::Text(s.clone()),
        tool_calls: None,
        tool_call_id: None,
    });

    let mut messages: Vec<OpenAIMessage> = req
        .messages
        .iter()
        .flat_map(ir_message_to_openai_messages)
        .collect();

    if let Some(sys) = system_msg {
        messages.insert(0, sys);
    }

    let tools: Vec<OpenAITool> = req
        .tools
        .iter()
        .map(|t| OpenAITool {
            ty: "function".into(),
            function: OpenAIFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            },
        })
        .collect();

    let tool_choice = req.tool_choice.as_ref().map(|tc| match tc {
        ToolChoice::Auto => OpenAIToolChoice::Str("auto".into()),
        ToolChoice::Any => OpenAIToolChoice::Str("required".into()),
        ToolChoice::None => OpenAIToolChoice::Str("none".into()),
        ToolChoice::Tool { name } => OpenAIToolChoice::Tool {
            ty: "function".into(),
            function: OpenAIToolChoiceFunction { name: name.clone() },
        },
    });

    OpenAIChatRequest {
        model: req.model.clone(),
        messages,
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        stop: req.stop_sequences.clone(),
        tools,
        tool_choice,
        stream: req.stream,
    }
}

fn ir_message_to_openai(msg: &Message) -> OpenAIMessage {
    let role = role_to_str(&msg.role);
    let (content, tool_calls, tool_call_id) = blocks_to_openai(&msg.content);
    // Guard: never send null/empty content to OpenAI-compatible backends.
    let content = match &content {
        OpenAIContent::Text(s)
            if s.is_empty() && tool_calls.is_none() && tool_call_id.is_none() =>
        {
            OpenAIContent::Text(" ".into())
        }
        _ => content,
    };
    OpenAIMessage {
        role,
        content,
        tool_calls,
        tool_call_id,
    }
}

fn ir_message_to_openai_messages(msg: &Message) -> Vec<OpenAIMessage> {
    let tool_results: Vec<OpenAIMessage> = msg
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolResult { id, content, .. } if !id.is_empty() => Some(OpenAIMessage {
                role: "tool".into(),
                content: OpenAIContent::Text(if content.is_empty() {
                    " ".into()
                } else {
                    content.clone()
                }),
                tool_calls: None,
                tool_call_id: Some(id.clone()),
            }),
            _ => None,
        })
        .collect();

    if tool_results.is_empty() {
        return vec![ir_message_to_openai(msg)];
    }

    let non_tool_blocks: Vec<ContentBlock> = msg
        .content
        .iter()
        .filter(|block| !matches!(block, ContentBlock::ToolResult { .. }))
        .cloned()
        .collect();

    let mut messages = Vec::new();
    if !non_tool_blocks.is_empty() {
        messages.push(ir_message_to_openai(&Message {
            role: msg.role.clone(),
            content: non_tool_blocks,
        }));
    }
    messages.extend(tool_results);
    messages
}

fn blocks_to_openai(
    blocks: &[ContentBlock],
) -> (OpenAIContent, Option<Vec<OpenAIToolCall>>, Option<String>) {
    // Check if it's pure text or multipart
    let text_only = blocks.len() == 1 && matches!(blocks[0], ContentBlock::Text { .. });
    let has_images = blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }));
    let has_tool_use = blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

    if has_tool_use {
        let tool_calls: Vec<OpenAIToolCall> = blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => Some(OpenAIToolCall {
                    id: id.clone(),
                    ty: "function".into(),
                    function: OpenAIFunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    },
                }),
                _ => None,
            })
            .collect();
        return (OpenAIContent::Text(String::new()), Some(tool_calls), None);
    }

    let tool_call_id = blocks.iter().find_map(|b| match b {
        ContentBlock::ToolResult { id, .. } => Some(id.clone()),
        _ => None,
    });

    if text_only {
        if let ContentBlock::Text { text } = &blocks[0] {
            return (OpenAIContent::Text(text.clone()), None, tool_call_id);
        }
    }

    if has_images || blocks.len() > 1 {
        let parts: Vec<OpenAIContentPart> = blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(OpenAIContentPart::Text { text: text.clone() }),
                ContentBlock::Image { source, media_type } => {
                    let url = match source {
                        ImageSource::Base64 { data } => {
                            format!("data:{};base64,{}", media_type, data)
                        }
                        ImageSource::Url { url } => url.clone(),
                    };
                    Some(OpenAIContentPart::ImageUrl {
                        image_url: OpenAIImageUrl { url },
                    })
                }
                _ => None,
            })
            .collect();
        if parts.is_empty() {
            return (OpenAIContent::Text(" ".into()), None, tool_call_id);
        }
        return (OpenAIContent::MultiPart(parts), None, tool_call_id);
    }

    (OpenAIContent::Text(" ".into()), None, None)
}

fn role_to_str(role: &Role) -> String {
    match role {
        Role::System => "system".into(),
        Role::User => "user".into(),
        Role::Assistant => "assistant".into(),
        Role::Tool => "tool".into(),
    }
}

fn openai_response_to_ir(resp: OpenAIResponse) -> ChatResponse {
    let choices: Vec<Choice> = resp
        .choices
        .into_iter()
        .map(|c| {
            let message = c.message.map(|m| {
                let content = openai_message_to_blocks(&m);
                Message {
                    role: str_to_role(m.role.as_deref()),
                    content,
                }
            });
            Choice {
                index: c.index,
                message,
                delta: None,
                finish_reason: c.finish_reason.as_deref().map(parse_finish_reason),
            }
        })
        .collect();

    let usage = resp
        .usage
        .map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        })
        .unwrap_or_default();

    ChatResponse {
        id: resp.id,
        model: resp.model,
        choices,
        usage,
    }
}

fn openai_message_to_blocks(msg: &OpenAIRespMessage) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();

    if let Some(text) = &msg.content {
        if !text.is_empty() {
            blocks.push(ContentBlock::Text { text: text.clone() });
        }
    }

    for tc in &msg.tool_calls {
        let input: Value = serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);
        blocks.push(ContentBlock::ToolUse {
            id: tc.id.clone(),
            name: tc.function.name.clone(),
            input,
        });
    }

    if blocks.is_empty() {
        blocks.push(ContentBlock::Text {
            text: String::new(),
        });
    }

    blocks
}

fn str_to_role(s: Option<&str>) -> Role {
    match s {
        Some("system") | Some("developer") => Role::System,
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        Some("tool") => Role::Tool,
        _ => Role::Assistant,
    }
}

fn parse_finish_reason(s: &str) -> FinishReason {
    match s {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

fn summarize_openai_request(req: &OpenAIChatRequest) -> String {
    let mut summary = format!("model={}, stream={}, messages=[", req.model, req.stream);
    for (index, msg) in req.messages.iter().enumerate() {
        if index > 0 {
            summary.push_str(", ");
        }
        let content = match &msg.content {
            OpenAIContent::Text(text) => format!("text(len={})", text.len()),
            OpenAIContent::MultiPart(parts) => format!("multipart(parts={})", parts.len()),
        };
        let tool_calls = msg
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls
                    .iter()
                    .map(|call| call.id.as_str())
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .unwrap_or_else(|| "-".into());
        let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("-");
        summary.push_str(&format!(
            "#{index}:role={},content={},tool_calls=[{}],tool_call_id={}",
            msg.role, content, tool_calls, tool_call_id
        ));
    }
    summary.push(']');
    summary
}

// --- Adapter implementation ---

#[async_trait]
impl Adapter for OpenaiAdapter {
    fn provider_name(&self) -> &str {
        "openai"
    }

    fn supports_model(&self, model: &str) -> bool {
        // OpenAI adapter accepts all models by default; routing happens upstream.
        !model.starts_with("claude-")
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, AdapterError> {
        let native = ir_to_openai_request(request);
        let url = format!("{}/v1/chat/completions", self.base_url);
        info!(provider = "openai", model = %request.model, stream = request.stream, "sending chat request");
        trace!(provider = "openai", url = %url, body_model = %native.model, "openai request prepared");
        trace!(provider = "openai", request = %summarize_openai_request(&native), "openai outbound request");

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
            error!(provider = "openai", status = %status, error = %body, "backend returned error");
            return Err(AdapterError::BackendError(format!(
                "OpenAI API returned error: {}",
                body
            )));
        }

        debug!(provider = "openai", status = %resp.status(), "backend response ok");
        let openai_resp: OpenAIResponse = resp
            .json()
            .await
            .map_err(|e| AdapterError::TranslationError(e.to_string()))?;

        Ok(openai_response_to_ir(openai_resp))
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, AdapterError>> + Send>>, AdapterError>
    {
        let mut native = ir_to_openai_request(request);
        native.stream = true;

        let url = format!("{}/v1/chat/completions", self.base_url);
        info!(provider = "openai", model = %request.model, stream = true, "sending streaming request");
        trace!(provider = "openai", request = %summarize_openai_request(&native), "openai outbound streaming request");

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
            error!(provider = "openai", status = %status, error = %body, "stream backend returned error");
            return Err(AdapterError::BackendError(format!(
                "OpenAI API returned error: {}",
                body
            )));
        }

        use futures::StreamExt;

        let byte_stream = resp.bytes_stream();
        let event_stream = futures::stream::unfold(
            (
                byte_stream,
                String::new(),
                VecDeque::<StreamEvent>::new(),
                OpenAIToolStreamState::default(),
            ),
            |(mut byte_stream, mut buffer, mut pending, mut tool_state)| async move {
                loop {
                    if let Some(event) = pending.pop_front() {
                        return Some((Ok(event), (byte_stream, buffer, pending, tool_state)));
                    }

                    // Yield complete lines from buffer
                    if let Some(pos) = buffer.find('\n') {
                        let line = buffer[..pos].trim().to_string();
                        buffer = buffer[pos + 1..].to_string();
                        if line.is_empty() {
                            continue;
                        }
                        if line == "data: [DONE]" {
                            return None; // stream end
                        }
                        if let Ok(events) = parse_openai_sse_line_events(&line, &mut tool_state) {
                            pending.extend(events);
                            if let Some(event) = pending.pop_front() {
                                return Some((
                                    Ok(event),
                                    (byte_stream, buffer, pending, tool_state),
                                ));
                            }
                        }
                        // unparseable line → skip
                        continue;
                    }

                    // Need more data
                    match byte_stream.next().await {
                        Some(Ok(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Some(Err(e)) => {
                            warn!(provider = "openai", error = %e, "stream byte read error");
                            return Some((
                                Err(AdapterError::StreamError(e.to_string())),
                                (byte_stream, buffer, pending, tool_state),
                            ));
                        }
                        None => return None,
                    }
                }
            },
        );

        Ok(Box::pin(event_stream))
    }
}

fn parse_openai_sse_line_events(
    line: &str,
    tool_state: &mut OpenAIToolStreamState,
) -> Result<Vec<StreamEvent>, AdapterError> {
    let json_str = line.strip_prefix("data: ").unwrap_or(line);
    trace!(provider = "openai", sse_line = %json_str, "parsing sse line");
    let chunk: OpenAISseChunk = serde_json::from_str(json_str)
        .map_err(|e| AdapterError::TranslationError(format!("failed to parse SSE chunk: {e}")))?;

    if let Some(choices) = chunk.choices {
        if let Some(choice) = choices.into_iter().next() {
            let index = choice.index;
            let finish_reason = choice.finish_reason.as_deref().map(parse_finish_reason);
            let mut events = Vec::new();

            if let Some(d) = choice.delta {
                if let Some(text) = d.content.filter(|s| !s.is_empty()) {
                    events.push(StreamEvent::ContentBlockDelta {
                        index,
                        delta: ContentDelta::TextDelta { text },
                    });
                }

                for tc in d.tool_calls {
                    let tool_index = tc.index;
                    if let Entry::Vacant(entry) = tool_state.started.entry(tool_index) {
                        let id = tc
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("tool_call_{tool_index}"));
                        let name = tc
                            .function
                            .as_ref()
                            .and_then(|func| func.name.clone())
                            .unwrap_or_default();
                        entry.insert(id.clone());
                        events.push(StreamEvent::ContentBlockStart {
                            index: tool_index,
                            content_block: ContentBlock::ToolUse {
                                id,
                                name,
                                input: Value::Object(Default::default()),
                            },
                        });
                    }

                    if let Some(arguments) = tc
                        .function
                        .and_then(|func| func.arguments)
                        .filter(|arguments| !arguments.is_empty())
                    {
                        events.push(StreamEvent::ContentBlockDelta {
                            index: tool_index,
                            delta: ContentDelta::InputJSONDelta {
                                partial_json: arguments,
                            },
                        });
                    }
                }
            }

            if let Some(reason) = &finish_reason {
                match reason {
                    FinishReason::ToolCalls => {
                        let mut indices: Vec<u32> = tool_state.started.keys().copied().collect();
                        indices.sort_unstable();
                        for tool_index in indices {
                            events.push(StreamEvent::ContentBlockStop { index: tool_index });
                        }
                        events.push(StreamEvent::MessageDelta {
                            stop_reason: Some("tool_use".into()),
                            usage: chunk.usage.map(|u| Usage {
                                prompt_tokens: u.prompt_tokens,
                                completion_tokens: u.completion_tokens,
                                total_tokens: u.total_tokens,
                            }),
                        });
                    }
                    FinishReason::Stop | FinishReason::Length => {
                        // Text block: emit ContentBlockStop for the current block.
                        events.push(StreamEvent::ContentBlockStop { index });
                        events.push(StreamEvent::MessageDelta {
                            stop_reason: Some("end_turn".into()),
                            usage: chunk.usage.map(|u| Usage {
                                prompt_tokens: u.prompt_tokens,
                                completion_tokens: u.completion_tokens,
                                total_tokens: u.total_tokens,
                            }),
                        });
                    }
                    FinishReason::ContentFilter => {
                        events.push(StreamEvent::Error {
                            code: "content_filter".into(),
                            message: "content filter triggered".into(),
                        });
                    }
                }
            }

            if !events.is_empty() {
                return Ok(events);
            }
        }
    }

    Err(AdapterError::StreamError("no choices in SSE chunk".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base_request(messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            model: "gpt-5.5".into(),
            messages,
            system: None,
            temperature: None,
            max_tokens: Some(128),
            stop_sequences: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            stream: false,
            extra: Default::default(),
        }
    }

    #[test]
    fn assistant_tool_uses_become_single_openai_tool_calls_message() {
        let req = base_request(vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::ToolUse {
                    id: "toolu_1".into(),
                    name: "read_file".into(),
                    input: json!({"path": "Cargo.toml"}),
                },
                ContentBlock::ToolUse {
                    id: "toolu_2".into(),
                    name: "shell".into(),
                    input: json!({"command": ["pwd"]}),
                },
            ],
        }]);

        let native = ir_to_openai_request(&req);
        let value = serde_json::to_value(native).expect("serialize request");
        let messages = value["messages"].as_array().expect("messages array");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "assistant");
        assert!(messages[0].get("content").is_none());

        let tool_calls = messages[0]["tool_calls"].as_array().expect("tool calls");
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0]["id"], "toolu_1");
        assert_eq!(tool_calls[0]["function"]["name"], "read_file");
        assert_eq!(
            tool_calls[0]["function"]["arguments"],
            r#"{"path":"Cargo.toml"}"#
        );
        assert_eq!(tool_calls[1]["id"], "toolu_2");
        assert_eq!(tool_calls[1]["function"]["name"], "shell");
        assert_eq!(
            tool_calls[1]["function"]["arguments"],
            r#"{"command":["pwd"]}"#
        );
    }

    #[test]
    fn tool_results_become_openai_tool_role_messages_with_matching_ids() {
        let req = base_request(vec![Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    id: "toolu_1".into(),
                    content: "file contents".into(),
                    is_error: false,
                },
                ContentBlock::ToolResult {
                    id: "toolu_2".into(),
                    content: "shell output".into(),
                    is_error: false,
                },
            ],
        }]);

        let native = ir_to_openai_request(&req);
        let value = serde_json::to_value(native).expect("serialize request");
        let messages = value["messages"].as_array().expect("messages array");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "toolu_1");
        assert_eq!(messages[0]["content"], "file contents");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "toolu_2");
        assert_eq!(messages[1]["content"], "shell output");
    }

    #[test]
    fn empty_tool_result_content_serializes_as_space_not_null() {
        let req = base_request(vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                id: "toolu_empty".into(),
                content: String::new(),
                is_error: false,
            }],
        }]);

        let native = ir_to_openai_request(&req);
        let value = serde_json::to_value(native).expect("serialize request");
        let messages = value["messages"].as_array().expect("messages array");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["tool_call_id"], "toolu_empty");
        assert_eq!(messages[0]["content"], " ");
    }

    #[test]
    fn empty_tool_result_id_is_not_forwarded_as_tool_message() {
        let req = base_request(vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                id: String::new(),
                content: "orphaned output".into(),
                is_error: false,
            }],
        }]);

        let native = ir_to_openai_request(&req);
        let value = serde_json::to_value(native).expect("serialize request");
        let messages = value["messages"].as_array().expect("messages array");

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], " ");
        assert!(messages[0].get("tool_call_id").is_none());
    }

    #[test]
    fn streaming_tool_call_delta_starts_tool_use_and_streams_arguments() {
        let mut state = OpenAIToolStreamState::default();
        let line = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\""}}]}}]}"#;

        let events = parse_openai_sse_line_events(line, &mut state).expect("parse events");

        assert_eq!(events.len(), 2);
        match &events[0] {
            StreamEvent::ContentBlockStart {
                index,
                content_block: ContentBlock::ToolUse { id, name, input },
            } => {
                assert_eq!(*index, 0);
                assert_eq!(id, "call_1");
                assert_eq!(name, "read_file");
                assert_eq!(input, &json!({}));
            }
            other => panic!("expected tool_use start, got {other:?}"),
        }
        match &events[1] {
            StreamEvent::ContentBlockDelta {
                index,
                delta: ContentDelta::InputJSONDelta { partial_json },
            } => {
                assert_eq!(*index, 0);
                assert_eq!(partial_json, r#"{"path""#);
            }
            other => panic!("expected input json delta, got {other:?}"),
        }
    }

    #[test]
    fn streaming_tool_call_delta_continues_arguments_without_restarting_block() {
        let mut state = OpenAIToolStreamState::default();
        parse_openai_sse_line_events(
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\""}}]}}]}"#,
            &mut state,
        )
        .expect("parse initial events");

        let events = parse_openai_sse_line_events(
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"Cargo.toml\"}"}}]}}]}"#,
            &mut state,
        )
        .expect("parse continuation events");

        assert_eq!(events.len(), 1);
        match &events[0] {
            StreamEvent::ContentBlockDelta {
                index,
                delta: ContentDelta::InputJSONDelta { partial_json },
            } => {
                assert_eq!(*index, 0);
                assert_eq!(partial_json, r#":"Cargo.toml"}"#);
            }
            other => panic!("expected input json delta, got {other:?}"),
        }
    }

    #[test]
    fn streaming_tool_call_finish_emits_tool_stop_and_message_delta() {
        let mut state = OpenAIToolStreamState::default();
        parse_openai_sse_line_events(
            r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{}"}}]}}]}"#,
            &mut state,
        )
        .expect("parse initial events");

        let events = parse_openai_sse_line_events(
            r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            &mut state,
        )
        .expect("parse finish events");

        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0],
            StreamEvent::ContentBlockStop { index: 0 }
        ));
        match &events[1] {
            StreamEvent::MessageDelta { stop_reason, usage } => {
                assert_eq!(stop_reason.as_deref(), Some("tool_use"));
                assert!(usage.is_none());
            }
            other => panic!("expected message delta, got {other:?}"),
        }
    }
}

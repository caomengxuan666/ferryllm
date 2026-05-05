use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{sse::Event, IntoResponse, Sse},
    routing::post,
    Router,
};
use futures::StreamExt;

use crate::adapter::AdapterError;
use crate::entry::{anthropic, openai};
use crate::ir;
use crate::router::{ResolvedFallback, Router as ModelRouter};
use tracing::{error, info, warn};

pub struct AppState {
    pub router: ModelRouter,
    pub options: ServerOptions,
    pub metrics: Metrics,
}

#[derive(Clone)]
pub struct ServerOptions {
    pub request_timeout_secs: u64,
    pub body_limit_bytes: usize,
    pub auth_enabled: bool,
    pub auth_keys: Vec<String>,
    pub metrics_enabled: bool,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            request_timeout_secs: 120,
            body_limit_bytes: 32 * 1024 * 1024,
            auth_enabled: false,
            auth_keys: Vec::new(),
            metrics_enabled: true,
        }
    }
}

#[derive(Default)]
pub struct Metrics {
    requests_total: AtomicU64,
    requests_ok_total: AtomicU64,
    requests_error_total: AtomicU64,
    upstream_errors_total: AtomicU64,
}

impl Metrics {
    fn inc_requests(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_ok(&self) {
        self.requests_ok_total.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_error(&self) {
        self.requests_error_total.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_upstream_error(&self) {
        self.upstream_errors_total.fetch_add(1, Ordering::Relaxed);
    }

    fn render(&self) -> String {
        format!(
            "# TYPE ferryllm_requests_total counter\nferryllm_requests_total {}\n# TYPE ferryllm_requests_ok_total counter\nferryllm_requests_ok_total {}\n# TYPE ferryllm_requests_error_total counter\nferryllm_requests_error_total {}\n# TYPE ferryllm_upstream_errors_total counter\nferryllm_upstream_errors_total {}\n",
            self.requests_total.load(Ordering::Relaxed),
            self.requests_ok_total.load(Ordering::Relaxed),
            self.requests_error_total.load(Ordering::Relaxed),
            self.upstream_errors_total.load(Ordering::Relaxed),
        )
    }
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(openai_chat_handler))
        .route("/v1/messages", post(anthropic_messages_handler))
        .route("/health", axum::routing::get(health_handler))
        .route("/healthz", axum::routing::get(health_handler))
        .route("/readyz", axum::routing::get(ready_handler))
        .route("/metrics", axum::routing::get(metrics_handler))
        .with_state(state)
}

// ── OpenAI-compatible endpoint ──────────────────────────────────────────────

async fn openai_chat_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, AppError> {
    preflight(&state, &headers, body.len())?;
    state.metrics.inc_requests();
    let request_id = request_id(&headers);
    let body: serde_json::Value =
        serde_json::from_slice(&body).map_err(|e| AppError::bad_request(e.to_string()))?;
    let openai_req: openai::OpenAIChatRequest =
        serde_json::from_value(body).map_err(|e| AppError::bad_request(e.to_string()))?;
    let stream = openai_req.stream;
    let mut ir_req = openai::openai_to_ir(&openai_req);
    info!(request_id = %request_id, entry = "openai", model = %ir_req.model, stream, "incoming request");
    let route = state
        .router
        .resolve(&ir_req.model)
        .map_err(AppError::from_adapter)?;

    // Apply model rewrite so the backend sees the correct model name.
    let display_model = ir_req.model.clone();
    ir_req.model = route.model;
    info!(request_id = %request_id, entry = "openai", display_model = %display_model, backend_model = %ir_req.model, provider = %route.provider, fallbacks = route.fallbacks.len(), "resolved route");

    if stream {
        handle_openai_stream(&state, route.adapter, ir_req, display_model).await
    } else {
        handle_openai_chat(
            &state,
            route.adapter,
            route.fallbacks,
            ir_req,
            display_model,
        )
        .await
    }
}

async fn handle_openai_chat(
    state: &Arc<AppState>,
    adapter: Arc<dyn crate::adapter::Adapter>,
    fallbacks: Vec<ResolvedFallback>,
    ir_req: ir::ChatRequest,
    display_model: String,
) -> Result<axum::response::Response, AppError> {
    let mut ir_resp = chat_with_fallbacks(state, adapter, fallbacks, &ir_req).await?;
    ir_resp.model = display_model;
    let openai_resp = openai::ir_to_openai_response(ir_resp);
    let body = serde_json::to_string(&openai_resp).unwrap_or_default();
    Ok((
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response())
}

async fn handle_openai_stream(
    state: &Arc<AppState>,
    adapter: Arc<dyn crate::adapter::Adapter>,
    ir_req: ir::ChatRequest,
    model: String,
) -> Result<axum::response::Response, AppError> {
    let backend_stream = with_timeout(state, adapter.chat_stream(&ir_req)).await?;

    let message_id = format!("chatcmpl-{}", uuid_v4());

    let sse_stream = backend_stream.filter_map(move |result| {
        let mid = message_id.clone();
        let m = model.clone();
        async move {
            match result {
                Ok(stream_event) => {
                    let line = openai::ir_to_openai_sse(stream_event, &mid, &m);
                    line.map(|data| {
                        Ok::<_, Infallible>(Event::default().data(openai_sse_data(&data)))
                    })
                }
                Err(e) => Some(Ok(
                    Event::default().data(format!("data: {{\"error\": \"{}\"}}\n", e))
                )),
            }
        }
    });

    Ok(Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response())
}

fn openai_sse_data(line: &str) -> String {
    line.trim()
        .strip_prefix("data: ")
        .unwrap_or(line.trim())
        .to_string()
}

// ── Anthropic-compatible endpoint ───────────────────────────────────────────

async fn anthropic_messages_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, AppError> {
    preflight(&state, &headers, body.len())?;
    state.metrics.inc_requests();
    let request_id = request_id(&headers);
    let body: serde_json::Value =
        serde_json::from_slice(&body).map_err(|e| AppError::bad_request(e.to_string()))?;
    let anthro_req: anthropic::AnthropicMessageRequest =
        serde_json::from_value(body).map_err(|e| AppError::bad_request(e.to_string()))?;
    let stream = anthro_req.stream;
    let mut ir_req = anthropic::anthropic_to_ir(&anthro_req);
    info!(request_id = %request_id, entry = "anthropic", model = %ir_req.model, stream, "incoming request");
    let route = state
        .router
        .resolve(&ir_req.model)
        .map_err(AppError::from_adapter)?;

    // Apply model rewrite.
    let display_model = ir_req.model.clone();
    ir_req.model = route.model;
    info!(request_id = %request_id, entry = "anthropic", display_model = %display_model, backend_model = %ir_req.model, provider = %route.provider, fallbacks = route.fallbacks.len(), "resolved route");

    if stream {
        handle_anthropic_stream(&state, route.adapter, ir_req, display_model).await
    } else {
        handle_anthropic_chat(
            &state,
            route.adapter,
            route.fallbacks,
            ir_req,
            display_model,
        )
        .await
    }
}

async fn handle_anthropic_chat(
    state: &Arc<AppState>,
    adapter: Arc<dyn crate::adapter::Adapter>,
    fallbacks: Vec<ResolvedFallback>,
    ir_req: ir::ChatRequest,
    display_model: String,
) -> Result<axum::response::Response, AppError> {
    let mut ir_resp = chat_with_fallbacks(state, adapter, fallbacks, &ir_req).await?;
    ir_resp.model = display_model;
    let anthro_resp = anthropic::ir_to_anthropic_response(ir_resp);
    let body = serde_json::to_string(&anthro_resp).unwrap_or_default();
    Ok((
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response())
}

async fn handle_anthropic_stream(
    state: &Arc<AppState>,
    adapter: Arc<dyn crate::adapter::Adapter>,
    ir_req: ir::ChatRequest,
    _display_model: String,
) -> Result<axum::response::Response, AppError> {
    let backend_stream = with_timeout(state, adapter.chat_stream(&ir_req)).await?;

    let sse_stream = backend_stream.filter_map(|result| async move {
        match result {
            Ok(stream_event) => {
                let sse = anthropic::ir_to_anthropic_sse(stream_event);
                sse.map(|(event_type, data)| {
                    Ok::<_, Infallible>(Event::default().event(event_type).data(data))
                })
            }
            Err(e) => Some(Ok(
                Event::default().data(format!("{{\"error\": \"{}\"}}", e))
            )),
        }
    });

    Ok(Sse::new(sse_stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response())
}

async fn chat_with_fallbacks(
    state: &Arc<AppState>,
    adapter: Arc<dyn crate::adapter::Adapter>,
    fallbacks: Vec<ResolvedFallback>,
    ir_req: &ir::ChatRequest,
) -> Result<ir::ChatResponse, AppError> {
    match with_timeout(state, adapter.chat(ir_req)).await {
        Ok(resp) => Ok(resp),
        Err(primary_error) if !fallbacks.is_empty() => {
            let mut last_error = primary_error;
            for fallback in fallbacks {
                let mut fallback_req = ir_req.clone();
                fallback_req.model = fallback.model;
                warn!(provider = %fallback.provider, model = %fallback_req.model, "trying fallback provider");
                match with_timeout(state, fallback.adapter.chat(&fallback_req)).await {
                    Ok(resp) => return Ok(resp),
                    Err(err) => last_error = err,
                }
            }
            Err(last_error)
        }
        Err(err) => Err(err),
    }
}

// ── Health ──────────────────────────────────────────────────────────────────

async fn health_handler() -> &'static str {
    "ok"
}

async fn ready_handler() -> &'static str {
    "ready"
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> Result<String, AppError> {
    if !state.options.metrics_enabled {
        return Err(AppError::not_found("metrics disabled".into()));
    }
    Ok(state.metrics.render())
}

fn preflight(state: &Arc<AppState>, headers: &HeaderMap, body_len: usize) -> Result<(), AppError> {
    if body_len > state.options.body_limit_bytes {
        state.metrics.inc_error();
        return Err(AppError::payload_too_large(format!(
            "request body too large: {} bytes",
            body_len
        )));
    }
    if state.options.auth_enabled && !is_authorized(&state.options.auth_keys, headers) {
        state.metrics.inc_error();
        return Err(AppError::unauthorized("invalid or missing API key".into()));
    }
    Ok(())
}

fn is_authorized(keys: &[String], headers: &HeaderMap) -> bool {
    let bearer = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    let x_api_key = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok());
    bearer
        .into_iter()
        .chain(x_api_key)
        .any(|candidate| keys.iter().any(|key| key == candidate))
}

fn request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(uuid_v4)
}

async fn with_timeout<T>(
    state: &Arc<AppState>,
    future: impl std::future::Future<Output = Result<T, AdapterError>>,
) -> Result<T, AppError> {
    match tokio::time::timeout(
        Duration::from_secs(state.options.request_timeout_secs),
        future,
    )
    .await
    {
        Ok(Ok(value)) => {
            state.metrics.inc_ok();
            Ok(value)
        }
        Ok(Err(err)) => {
            state.metrics.inc_upstream_error();
            state.metrics.inc_error();
            Err(AppError::from_adapter(err))
        }
        Err(_) => {
            state.metrics.inc_upstream_error();
            state.metrics.inc_error();
            Err(AppError::timeout("upstream request timed out".into()))
        }
    }
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:032x}", ts)
}

// ── Error handling ──────────────────────────────────────────────────────────

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(msg: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg,
        }
    }

    fn unauthorized(msg: String) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: msg,
        }
    }

    fn payload_too_large(msg: String) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: msg,
        }
    }

    fn timeout(msg: String) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: msg,
        }
    }

    fn not_found(msg: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg,
        }
    }

    fn from_adapter(err: AdapterError) -> Self {
        let status = match &err {
            AdapterError::BackendError(_) => StatusCode::BAD_GATEWAY,
            AdapterError::TranslationError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AdapterError::StreamError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AdapterError::UnsupportedFeature { .. } => StatusCode::NOT_IMPLEMENTED,
        };
        Self {
            status,
            message: err.to_string(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        error!(status = %self.status, error = %self.message, "request failed");
        let body = serde_json::json!({"error": {"message": self.message}});
        (self.status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::stream;

    struct TestAdapter {
        name: &'static str,
        fail: bool,
    }

    #[async_trait]
    impl crate::adapter::Adapter for TestAdapter {
        fn provider_name(&self) -> &str {
            self.name
        }

        fn supports_model(&self, _model: &str) -> bool {
            true
        }

        async fn chat(&self, request: &ir::ChatRequest) -> Result<ir::ChatResponse, AdapterError> {
            if self.fail {
                return Err(AdapterError::BackendError("primary failed".into()));
            }
            Ok(ir::ChatResponse {
                id: "resp_1".into(),
                model: request.model.clone(),
                choices: vec![ir::Choice {
                    index: 0,
                    message: Some(ir::Message {
                        role: ir::Role::Assistant,
                        content: vec![ir::ContentBlock::Text {
                            text: self.name.into(),
                        }],
                    }),
                    delta: None,
                    finish_reason: Some(ir::FinishReason::Stop),
                }],
                usage: ir::Usage::default(),
            })
        }

        async fn chat_stream(
            &self,
            _request: &ir::ChatRequest,
        ) -> Result<
            std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<ir::StreamEvent, AdapterError>> + Send>,
            >,
            AdapterError,
        > {
            Ok(Box::pin(stream::empty()))
        }
    }

    #[tokio::test]
    async fn chat_with_fallbacks_returns_backup_response_after_primary_failure() {
        let state = Arc::new(AppState {
            router: ModelRouter::new(),
            options: ServerOptions::default(),
            metrics: Metrics::default(),
        });
        let primary = Arc::new(TestAdapter {
            name: "primary",
            fail: true,
        });
        let fallback = ResolvedFallback {
            adapter: Arc::new(TestAdapter {
                name: "backup",
                fail: false,
            }),
            provider: "backup".into(),
            model: "backup-model".into(),
        };
        let request = ir::ChatRequest {
            model: "primary-model".into(),
            messages: Vec::new(),
            system: None,
            temperature: None,
            max_tokens: None,
            stop_sequences: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            stream: false,
            extra: Default::default(),
        };

        let response = chat_with_fallbacks(&state, primary, vec![fallback], &request)
            .await
            .expect("fallback response");

        assert_eq!(response.model, "backup-model");
        assert_eq!(
            state.metrics.upstream_errors_total.load(Ordering::Relaxed),
            1
        );
    }
}

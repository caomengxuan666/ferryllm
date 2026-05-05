use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::{error, info, warn};

pub struct AppState {
    pub router: ModelRouter,
    pub options: ServerOptions,
    pub metrics: Metrics,
    concurrency: Option<Arc<Semaphore>>,
}

impl AppState {
    pub fn new(router: ModelRouter, options: ServerOptions, metrics: Metrics) -> Self {
        let concurrency = options
            .max_concurrent_requests
            .filter(|limit| *limit > 0)
            .map(Semaphore::new)
            .map(Arc::new);
        Self {
            router,
            options,
            metrics,
            concurrency,
        }
    }
}

#[derive(Clone)]
pub struct ServerOptions {
    pub request_timeout_secs: u64,
    pub body_limit_bytes: usize,
    pub max_concurrent_requests: Option<usize>,
    pub auth_enabled: bool,
    pub auth_keys: Vec<String>,
    pub metrics_enabled: bool,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            request_timeout_secs: 120,
            body_limit_bytes: 32 * 1024 * 1024,
            max_concurrent_requests: None,
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
    request_latency_micros_total: AtomicU64,
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

    fn observe_latency(&self, elapsed: Duration) {
        self.request_latency_micros_total.fetch_add(
            elapsed.as_micros().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
    }

    fn render(&self) -> String {
        let requests_total = self.requests_total.load(Ordering::Relaxed);
        let latency_total = self.request_latency_micros_total.load(Ordering::Relaxed);
        let latency_average = latency_total.checked_div(requests_total).unwrap_or(0);
        format!(
            "# TYPE ferryllm_requests_total counter\nferryllm_requests_total {}\n# TYPE ferryllm_requests_ok_total counter\nferryllm_requests_ok_total {}\n# TYPE ferryllm_requests_error_total counter\nferryllm_requests_error_total {}\n# TYPE ferryllm_upstream_errors_total counter\nferryllm_upstream_errors_total {}\n# TYPE ferryllm_request_latency_micros_total counter\nferryllm_request_latency_micros_total {}\n# TYPE ferryllm_request_latency_micros_average gauge\nferryllm_request_latency_micros_average {}\n",
            requests_total,
            self.requests_ok_total.load(Ordering::Relaxed),
            self.requests_error_total.load(Ordering::Relaxed),
            self.upstream_errors_total.load(Ordering::Relaxed),
            latency_total,
            latency_average,
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
    let started_at = Instant::now();
    preflight(&state, &headers, body.len())?;
    let _permit = acquire_concurrency(&state)?;
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
    let fallback_count = route.fallbacks.len();
    info!(request_id = %request_id, entry = "openai", display_model = %display_model, backend_model = %ir_req.model, provider = %route.provider, fallbacks = fallback_count, "resolved route");

    let response = if stream {
        handle_openai_stream(&state, route.adapter, ir_req, display_model.clone()).await
    } else {
        handle_openai_chat(
            &state,
            route.adapter,
            route.fallbacks,
            ir_req,
            display_model.clone(),
        )
        .await
    };
    state.metrics.observe_latency(started_at.elapsed());
    match response {
        Ok(response) => {
            log_access(RequestLog {
                entry: "openai",
                request_id: &request_id,
                model: &display_model,
                provider: &route.provider,
                stream,
                status: StatusCode::OK,
                latency: started_at.elapsed(),
                fallback_count,
                error_kind: None,
            });
            Ok(response)
        }
        Err(err) => {
            log_access(RequestLog {
                entry: "openai",
                request_id: &request_id,
                model: &display_model,
                provider: &route.provider,
                stream,
                status: err.status,
                latency: started_at.elapsed(),
                fallback_count,
                error_kind: Some(err.kind),
            });
            Err(err)
        }
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
    let started_at = Instant::now();
    preflight(&state, &headers, body.len())?;
    let _permit = acquire_concurrency(&state)?;
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
    let fallback_count = route.fallbacks.len();
    info!(request_id = %request_id, entry = "anthropic", display_model = %display_model, backend_model = %ir_req.model, provider = %route.provider, fallbacks = fallback_count, "resolved route");

    let response = if stream {
        handle_anthropic_stream(&state, route.adapter, ir_req, display_model.clone()).await
    } else {
        handle_anthropic_chat(
            &state,
            route.adapter,
            route.fallbacks,
            ir_req,
            display_model.clone(),
        )
        .await
    };
    state.metrics.observe_latency(started_at.elapsed());
    match response {
        Ok(response) => {
            log_access(RequestLog {
                entry: "anthropic",
                request_id: &request_id,
                model: &display_model,
                provider: &route.provider,
                stream,
                status: StatusCode::OK,
                latency: started_at.elapsed(),
                fallback_count,
                error_kind: None,
            });
            Ok(response)
        }
        Err(err) => {
            log_access(RequestLog {
                entry: "anthropic",
                request_id: &request_id,
                model: &display_model,
                provider: &route.provider,
                stream,
                status: err.status,
                latency: started_at.elapsed(),
                fallback_count,
                error_kind: Some(err.kind),
            });
            Err(err)
        }
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
    info!(provider = %adapter.provider_name(), model = %ir_req.model, fallback_count = fallbacks.len(), "attempting primary upstream request");
    match with_timeout(state, adapter.chat(ir_req)).await {
        Ok(resp) => Ok(resp),
        Err(primary_error) if !fallbacks.is_empty() => {
            warn!(provider = %adapter.provider_name(), error_kind = primary_error.kind, status = %primary_error.status, "primary upstream request failed");
            let mut last_error = primary_error;
            for fallback in fallbacks {
                let mut fallback_req = ir_req.clone();
                fallback_req.model = fallback.model;
                warn!(provider = %fallback.provider, model = %fallback_req.model, "trying fallback provider");
                match with_timeout(state, fallback.adapter.chat(&fallback_req)).await {
                    Ok(resp) => return Ok(resp),
                    Err(err) => {
                        warn!(provider = %fallback.provider, error_kind = err.kind, status = %err.status, "fallback upstream request failed");
                        last_error = err
                    }
                }
            }
            Err(last_error)
        }
        Err(err) => Err(err),
    }
}

struct RequestLog<'a> {
    entry: &'a str,
    request_id: &'a str,
    model: &'a str,
    provider: &'a str,
    stream: bool,
    status: StatusCode,
    latency: Duration,
    fallback_count: usize,
    error_kind: Option<&'a str>,
}

fn log_access(log: RequestLog<'_>) {
    info!(
        request_id = %log.request_id,
        entry = %log.entry,
        model = %log.model,
        provider = %log.provider,
        stream = log.stream,
        status = %log.status,
        latency_ms = log.latency.as_millis(),
        fallback_count = log.fallback_count,
        error_kind = log.error_kind.unwrap_or("none"),
        "request completed"
    );
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

fn acquire_concurrency(state: &Arc<AppState>) -> Result<Option<OwnedSemaphorePermit>, AppError> {
    let Some(limit) = state.concurrency.as_ref() else {
        return Ok(None);
    };
    limit.clone().try_acquire_owned().map(Some).map_err(|_| {
        state.metrics.inc_error();
        AppError::too_many_requests("too many concurrent requests".into())
    })
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
    kind: &'static str,
}

impl AppError {
    fn bad_request(msg: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg,
            kind: "bad_request",
        }
    }

    fn unauthorized(msg: String) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: msg,
            kind: "unauthorized",
        }
    }

    fn payload_too_large(msg: String) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: msg,
            kind: "payload_too_large",
        }
    }

    fn too_many_requests(msg: String) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: msg,
            kind: "too_many_requests",
        }
    }

    fn timeout(msg: String) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: msg,
            kind: "timeout",
        }
    }

    fn not_found(msg: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg,
            kind: "not_found",
        }
    }

    fn from_adapter(err: AdapterError) -> Self {
        let (status, kind) = match &err {
            AdapterError::BackendError(_) => (StatusCode::BAD_GATEWAY, "upstream_backend"),
            AdapterError::TranslationError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "translation"),
            AdapterError::StreamError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "stream"),
            AdapterError::UnsupportedFeature { .. } => {
                (StatusCode::NOT_IMPLEMENTED, "unsupported_feature")
            }
        };
        Self {
            status,
            message: err.to_string(),
            kind,
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
        let state = Arc::new(AppState::new(
            ModelRouter::new(),
            ServerOptions::default(),
            Metrics::default(),
        ));
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

    #[test]
    fn metrics_render_includes_latency_totals_and_average() {
        let metrics = Metrics::default();
        metrics.inc_requests();
        metrics.inc_requests();
        metrics.observe_latency(Duration::from_micros(100));
        metrics.observe_latency(Duration::from_micros(300));

        let rendered = metrics.render();

        assert!(rendered.contains("ferryllm_request_latency_micros_total 400"));
        assert!(rendered.contains("ferryllm_request_latency_micros_average 200"));
    }

    #[test]
    fn adapter_error_mapping_sets_status_and_kind() {
        let cases = [
            (
                AdapterError::BackendError("x".into()),
                StatusCode::BAD_GATEWAY,
                "upstream_backend",
            ),
            (
                AdapterError::TranslationError("x".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "translation",
            ),
            (
                AdapterError::StreamError("x".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "stream",
            ),
            (
                AdapterError::UnsupportedFeature {
                    provider: "p".into(),
                    feature: "f".into(),
                },
                StatusCode::NOT_IMPLEMENTED,
                "unsupported_feature",
            ),
        ];

        for (err, status, kind) in cases {
            let mapped = AppError::from_adapter(err);
            assert_eq!(mapped.status, status);
            assert_eq!(mapped.kind, kind);
        }
    }

    #[test]
    fn concurrency_limit_rejects_when_no_permits_remain() {
        let state = Arc::new(AppState::new(
            ModelRouter::new(),
            ServerOptions {
                max_concurrent_requests: Some(1),
                ..ServerOptions::default()
            },
            Metrics::default(),
        ));

        let first = acquire_concurrency(&state).expect("first permit");
        let second = acquire_concurrency(&state).expect_err("second permit should fail");

        assert!(first.is_some());
        assert_eq!(second.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(second.kind, "too_many_requests");
        assert_eq!(
            state.metrics.requests_error_total.load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn concurrency_limit_releases_when_permit_drops() {
        let state = Arc::new(AppState::new(
            ModelRouter::new(),
            ServerOptions {
                max_concurrent_requests: Some(1),
                ..ServerOptions::default()
            },
            Metrics::default(),
        ));

        let first = acquire_concurrency(&state).expect("first permit");
        drop(first);

        let second = acquire_concurrency(&state).expect("second permit after drop");

        assert!(second.is_some());
    }
}

use std::collections::HashMap;
use std::convert::Infallible;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    rate_limiter: Option<RateLimiter>,
    key_limiters: Mutex<HashMap<u64, Arc<KeyLimiter>>>,
    provider_health: Mutex<HashMap<String, Arc<ProviderHealth>>>,
}

impl AppState {
    pub fn new(router: ModelRouter, options: ServerOptions, metrics: Metrics) -> Self {
        let concurrency = options
            .max_concurrent_requests
            .filter(|limit| *limit > 0)
            .map(Semaphore::new)
            .map(Arc::new);
        let rate_limiter = options
            .rate_limit_per_minute
            .filter(|limit| *limit > 0)
            .map(RateLimiter::new);
        Self {
            router,
            options,
            metrics,
            concurrency,
            rate_limiter,
            key_limiters: Mutex::new(HashMap::new()),
            provider_health: Mutex::new(HashMap::new()),
        }
    }
}

#[derive(Clone)]
pub struct ServerOptions {
    pub request_timeout_secs: u64,
    pub body_limit_bytes: usize,
    pub max_concurrent_requests: Option<usize>,
    pub rate_limit_per_minute: Option<u64>,
    pub retry_attempts: u32,
    pub retry_backoff_ms: u64,
    pub circuit_breaker_failures: Option<u64>,
    pub circuit_breaker_cooldown_secs: u64,
    pub auth_enabled: bool,
    pub auth_keys: Vec<String>,
    pub per_key_rate_limit_per_minute: Option<u64>,
    pub per_key_max_concurrent_requests: Option<usize>,
    pub metrics_enabled: bool,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            request_timeout_secs: 120,
            body_limit_bytes: 32 * 1024 * 1024,
            max_concurrent_requests: None,
            rate_limit_per_minute: None,
            retry_attempts: 0,
            retry_backoff_ms: 100,
            circuit_breaker_failures: None,
            circuit_breaker_cooldown_secs: 30,
            auth_enabled: false,
            auth_keys: Vec::new(),
            per_key_rate_limit_per_minute: None,
            per_key_max_concurrent_requests: None,
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

struct RateLimiter {
    limit: u64,
    window_epoch_minute: AtomicU64,
    used: AtomicU64,
}

struct KeyLimiter {
    rate_limiter: Option<RateLimiter>,
    concurrency: Option<Arc<Semaphore>>,
}

struct ProviderHealth {
    consecutive_failures: AtomicU64,
    circuit_opened_at_epoch_secs: AtomicU64,
}

impl Default for ProviderHealth {
    fn default() -> Self {
        Self {
            consecutive_failures: AtomicU64::new(0),
            circuit_opened_at_epoch_secs: AtomicU64::new(0),
        }
    }
}

#[derive(Debug)]
struct RequestGuards {
    _global_concurrency: Option<OwnedSemaphorePermit>,
    _key_concurrency: Option<OwnedSemaphorePermit>,
}

impl RateLimiter {
    fn new(limit: u64) -> Self {
        Self {
            limit,
            window_epoch_minute: AtomicU64::new(current_epoch_minute()),
            used: AtomicU64::new(0),
        }
    }

    fn try_acquire(&self) -> bool {
        self.try_acquire_at(current_epoch_minute())
    }

    fn try_acquire_at(&self, epoch_minute: u64) -> bool {
        let current_window = self.window_epoch_minute.load(Ordering::Relaxed);
        if current_window != epoch_minute
            && self
                .window_epoch_minute
                .compare_exchange(
                    current_window,
                    epoch_minute,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
        {
            self.used.store(0, Ordering::Relaxed);
        }

        loop {
            let used = self.used.load(Ordering::Relaxed);
            if used >= self.limit {
                return false;
            }
            if self
                .used
                .compare_exchange(used, used + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
        }
    }
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
    let _guards = preflight(&state, &headers, body.len())?;
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
    let _guards = preflight(&state, &headers, body.len())?;
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

    let message_id = format!("msg_{}", uuid_v4());

    // Prepend MessageStart, then lift adapter stream items into the same
    // event type so they can be chained.
    enum Control {
        Injected(ir::StreamEvent),
        Adapter(Result<ir::StreamEvent, AdapterError>),
    }

    let prepend = futures::stream::once(async move {
        Control::Injected(ir::StreamEvent::MessageStart { message_id })
    });

    let adapter_stream = backend_stream.map(Control::Adapter);

    let combined = prepend.chain(adapter_stream);

    // State carried across stream items.
    use futures::StreamExt;
    use std::collections::HashSet;
    use std::sync::atomic::AtomicBool;
    let mut started_text_blocks: HashSet<u32> = HashSet::new();
    let closing_sent = Arc::new(AtomicBool::new(false));

    let closing_flag = Arc::clone(&closing_sent);
    let sse_stream = combined
        .map(move |control| {
            let mut pending: Vec<ir::StreamEvent> = Vec::new();

            let event = match control {
                Control::Injected(e) => e,
                Control::Adapter(Ok(e)) => e,
                Control::Adapter(Err(e)) => {
                    pending.push(ir::StreamEvent::Error {
                        code: "stream_error".into(),
                        message: e.to_string(),
                    });
                    return pending
                        .into_iter()
                        .filter_map(|e| {
                            let sse = anthropic::ir_to_anthropic_sse(e);
                            sse.map(|(event_type, data)| {
                                Ok::<_, Infallible>(Event::default().event(event_type).data(data))
                            })
                        })
                        .collect::<Vec<_>>();
                }
            };

            // Inject ContentBlockStart before first text delta.
            if let ir::StreamEvent::ContentBlockDelta {
                index,
                delta: ir::ContentDelta::TextDelta { .. },
            } = &event
            {
                if !started_text_blocks.contains(index) {
                    started_text_blocks.insert(*index);
                    pending.push(ir::StreamEvent::ContentBlockStart {
                        index: *index,
                        content_block: ir::ContentBlock::Text {
                            text: String::new(),
                        },
                    });
                }
            }

            if let ir::StreamEvent::MessageDelta { .. } | ir::StreamEvent::MessageStop = &event {
                closing_flag.store(true, Ordering::Relaxed);
            }
            pending.push(event);

            pending
                .into_iter()
                .filter_map(|e| {
                    let sse = anthropic::ir_to_anthropic_sse(e);
                    sse.map(|(event_type, data)| {
                        Ok::<_, Infallible>(Event::default().event(event_type).data(data))
                    })
                })
                .collect::<Vec<_>>()
        })
        // .map returns a stream of Vec<Result<Event, Infallible>>.
        // flat_map with stream::iter flattens each vec into individual items.
        .flat_map(|events: Vec<Result<Event, Infallible>>| futures::stream::iter(events));

    // Append MessageStop if the adapter stream ended without one.
    let tail = futures::stream::once(async move {
        if !closing_sent.load(Ordering::Relaxed) {
            let sse = anthropic::ir_to_anthropic_sse(ir::StreamEvent::MessageStop);
            sse.map(|(event_type, data)| {
                Ok::<_, Infallible>(Event::default().event(event_type).data(data))
            })
        } else {
            None
        }
    })
    .filter_map(futures::future::ready);

    Ok(Sse::new(sse_stream.chain(tail))
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
    match chat_with_resilience(state, &*adapter, ir_req).await {
        Ok(resp) => Ok(resp),
        Err(primary_error) if !fallbacks.is_empty() => {
            warn!(provider = %adapter.provider_name(), error_kind = primary_error.kind, status = %primary_error.status, "primary upstream request failed");
            let mut last_error = primary_error;
            for fallback in fallbacks {
                let mut fallback_req = ir_req.clone();
                fallback_req.model = fallback.model;
                warn!(provider = %fallback.provider, model = %fallback_req.model, "trying fallback provider");
                match chat_with_resilience(state, &*fallback.adapter, &fallback_req).await {
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

async fn chat_with_resilience(
    state: &Arc<AppState>,
    adapter: &dyn crate::adapter::Adapter,
    ir_req: &ir::ChatRequest,
) -> Result<ir::ChatResponse, AppError> {
    let provider = adapter.provider_name();
    let health = provider_health(state, provider);
    if circuit_is_open(state, provider, &health) {
        state.metrics.inc_error();
        return Err(AppError::service_unavailable(format!(
            "provider circuit open: {provider}"
        )));
    }

    let attempts = state.options.retry_attempts.saturating_add(1);
    let mut last_error = None;
    for attempt in 0..attempts {
        match with_timeout(state, adapter.chat(ir_req)).await {
            Ok(resp) => {
                record_provider_success(&health);
                return Ok(resp);
            }
            Err(err) => {
                let retryable = err.is_retryable();
                warn!(provider, attempt = attempt + 1, attempts, retryable, error_kind = err.kind, status = %err.status, "upstream request attempt failed");
                last_error = Some(err);
                if !retryable || attempt + 1 >= attempts {
                    break;
                }
                tokio::time::sleep(retry_delay(state, attempt)).await;
            }
        }
    }

    let err = last_error.unwrap_or_else(|| AppError::bad_gateway("upstream request failed".into()));
    record_provider_failure(state, provider, &health);
    Err(err)
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

fn provider_health(state: &Arc<AppState>, provider: &str) -> Arc<ProviderHealth> {
    let mut providers = state
        .provider_health
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Arc::clone(
        providers
            .entry(provider.to_string())
            .or_insert_with(|| Arc::new(ProviderHealth::default())),
    )
}

fn circuit_is_open(state: &Arc<AppState>, provider: &str, health: &ProviderHealth) -> bool {
    let Some(_) = state
        .options
        .circuit_breaker_failures
        .filter(|limit| *limit > 0)
    else {
        return false;
    };
    let opened_at = health.circuit_opened_at_epoch_secs.load(Ordering::Relaxed);
    if opened_at == 0 {
        return false;
    }
    let now = current_epoch_secs();
    let cooldown = state.options.circuit_breaker_cooldown_secs;
    if now.saturating_sub(opened_at) >= cooldown {
        warn!(provider, "provider circuit half-open after cooldown");
        return false;
    }
    true
}

fn record_provider_success(health: &ProviderHealth) {
    health.consecutive_failures.store(0, Ordering::Relaxed);
    health
        .circuit_opened_at_epoch_secs
        .store(0, Ordering::Relaxed);
}

fn record_provider_failure(state: &Arc<AppState>, provider: &str, health: &ProviderHealth) {
    let failures = health.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
    if let Some(limit) = state
        .options
        .circuit_breaker_failures
        .filter(|limit| *limit > 0)
    {
        if failures >= limit {
            health
                .circuit_opened_at_epoch_secs
                .store(current_epoch_secs(), Ordering::Relaxed);
            warn!(provider, failures, limit, "provider circuit opened");
        }
    }
}

fn retry_delay(state: &Arc<AppState>, attempt: u32) -> Duration {
    let multiplier = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
    Duration::from_millis(state.options.retry_backoff_ms.saturating_mul(multiplier))
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

fn preflight(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    body_len: usize,
) -> Result<RequestGuards, AppError> {
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
    let key_concurrency = if state.options.auth_enabled {
        authenticated_key(&state.options.auth_keys, headers)
            .map(|raw_key| acquire_key_limits(state, raw_key))
            .transpose()?
            .flatten()
    } else {
        None
    };
    if let Some(rate_limiter) = &state.rate_limiter {
        if !rate_limiter.try_acquire() {
            state.metrics.inc_error();
            return Err(AppError::rate_limited("rate limit exceeded".into()));
        }
    }
    let global_concurrency = acquire_concurrency(state)?;
    Ok(RequestGuards {
        _global_concurrency: global_concurrency,
        _key_concurrency: key_concurrency,
    })
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

fn acquire_key_limits(
    state: &Arc<AppState>,
    raw_key: &str,
) -> Result<Option<OwnedSemaphorePermit>, AppError> {
    let key_hash = hash_key(raw_key);
    let limiter = {
        let mut limiters = state
            .key_limiters
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Arc::clone(limiters.entry(key_hash).or_insert_with(|| {
            Arc::new(KeyLimiter {
                rate_limiter: state
                    .options
                    .per_key_rate_limit_per_minute
                    .filter(|limit| *limit > 0)
                    .map(RateLimiter::new),
                concurrency: state
                    .options
                    .per_key_max_concurrent_requests
                    .filter(|limit| *limit > 0)
                    .map(Semaphore::new)
                    .map(Arc::new),
            })
        }))
    };

    if let Some(rate_limiter) = &limiter.rate_limiter {
        if !rate_limiter.try_acquire() {
            state.metrics.inc_error();
            return Err(AppError::rate_limited("per-key rate limit exceeded".into()));
        }
    }

    if let Some(concurrency) = &limiter.concurrency {
        concurrency
            .clone()
            .try_acquire_owned()
            .map(Some)
            .map_err(|_| {
                state.metrics.inc_error();
                AppError::too_many_requests("per-key concurrency limit exceeded".into())
            })
    } else {
        Ok(None)
    }
}

fn hash_key(key: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

fn authenticated_key<'a>(keys: &'a [String], headers: &'a HeaderMap) -> Option<&'a str> {
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
        .find(|candidate| keys.iter().any(|key| key == *candidate))
}

fn is_authorized(keys: &[String], headers: &HeaderMap) -> bool {
    authenticated_key(keys, headers).is_some()
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
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:032x}", ts)
}

fn current_epoch_minute() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        / 60
}

fn current_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
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

    fn bad_gateway(msg: String) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: msg,
            kind: "upstream_backend",
        }
    }

    fn service_unavailable(msg: String) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: msg,
            kind: "circuit_open",
        }
    }

    fn rate_limited(msg: String) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: msg,
            kind: "rate_limited",
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

impl AppError {
    fn is_retryable(&self) -> bool {
        matches!(
            self.status,
            StatusCode::TOO_MANY_REQUESTS
                | StatusCode::BAD_GATEWAY
                | StatusCode::SERVICE_UNAVAILABLE
                | StatusCode::GATEWAY_TIMEOUT
        )
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

    struct FlakyAdapter {
        name: &'static str,
        failures_before_success: AtomicU64,
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

    #[async_trait]
    impl crate::adapter::Adapter for FlakyAdapter {
        fn provider_name(&self) -> &str {
            self.name
        }

        fn supports_model(&self, _model: &str) -> bool {
            true
        }

        async fn chat(&self, request: &ir::ChatRequest) -> Result<ir::ChatResponse, AdapterError> {
            let remaining = self.failures_before_success.load(Ordering::Relaxed);
            if remaining > 0 {
                self.failures_before_success.fetch_sub(1, Ordering::Relaxed);
                return Err(AdapterError::BackendError("temporary failure".into()));
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

    #[tokio::test]
    async fn retry_succeeds_after_temporary_backend_failure() {
        let state = Arc::new(AppState::new(
            ModelRouter::new(),
            ServerOptions {
                retry_attempts: 1,
                retry_backoff_ms: 0,
                circuit_breaker_failures: Some(3),
                ..ServerOptions::default()
            },
            Metrics::default(),
        ));
        let adapter = FlakyAdapter {
            name: "flaky",
            failures_before_success: AtomicU64::new(1),
        };
        let request = test_chat_request("model");

        let response = chat_with_resilience(&state, &adapter, &request)
            .await
            .expect("retry should recover");

        assert_eq!(response.model, "model");
        assert_eq!(adapter.failures_before_success.load(Ordering::Relaxed), 0);
        assert_eq!(
            provider_health(&state, "flaky")
                .consecutive_failures
                .load(Ordering::Relaxed),
            0
        );
    }

    #[tokio::test]
    async fn circuit_opens_after_consecutive_provider_failures() {
        let state = Arc::new(AppState::new(
            ModelRouter::new(),
            ServerOptions {
                circuit_breaker_failures: Some(1),
                circuit_breaker_cooldown_secs: 30,
                ..ServerOptions::default()
            },
            Metrics::default(),
        ));
        let adapter = TestAdapter {
            name: "broken",
            fail: true,
        };
        let request = test_chat_request("model");

        let first = chat_with_resilience(&state, &adapter, &request)
            .await
            .expect_err("first failure should open circuit");
        let second = chat_with_resilience(&state, &adapter, &request)
            .await
            .expect_err("open circuit should fail fast");

        assert_eq!(first.kind, "upstream_backend");
        assert_eq!(second.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(second.kind, "circuit_open");
    }

    #[tokio::test]
    async fn fallback_can_handle_open_primary_circuit() {
        let state = Arc::new(AppState::new(
            ModelRouter::new(),
            ServerOptions {
                circuit_breaker_failures: Some(1),
                circuit_breaker_cooldown_secs: 30,
                ..ServerOptions::default()
            },
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
        let request = test_chat_request("primary-model");

        let response = chat_with_fallbacks(&state, primary, vec![fallback], &request)
            .await
            .expect("fallback response");

        assert_eq!(response.model, "backup-model");
    }

    fn test_chat_request(model: &str) -> ir::ChatRequest {
        ir::ChatRequest {
            model: model.into(),
            messages: Vec::new(),
            system: None,
            temperature: None,
            max_tokens: None,
            stop_sequences: Vec::new(),
            tools: Vec::new(),
            tool_choice: None,
            stream: false,
            extra: Default::default(),
        }
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

    #[test]
    fn rate_limiter_rejects_after_limit_until_next_window() {
        let limiter = RateLimiter::new(2);

        assert!(limiter.try_acquire_at(100));
        assert!(limiter.try_acquire_at(100));
        assert!(!limiter.try_acquire_at(100));
        assert!(limiter.try_acquire_at(101));
    }

    #[test]
    fn preflight_rejects_when_rate_limit_exceeded() {
        let state = Arc::new(AppState::new(
            ModelRouter::new(),
            ServerOptions {
                rate_limit_per_minute: Some(1),
                ..ServerOptions::default()
            },
            Metrics::default(),
        ));
        let headers = HeaderMap::new();

        preflight(&state, &headers, 0).expect("first request");
        let err = preflight(&state, &headers, 0).expect_err("second request should fail");

        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.kind, "rate_limited");
        assert_eq!(
            state.metrics.requests_error_total.load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn per_key_rate_limit_isolated_by_api_key() {
        let state = Arc::new(AppState::new(
            ModelRouter::new(),
            ServerOptions {
                auth_enabled: true,
                auth_keys: vec!["key-a".into(), "key-b".into()],
                per_key_rate_limit_per_minute: Some(1),
                ..ServerOptions::default()
            },
            Metrics::default(),
        ));
        let headers_a = headers_with_bearer("key-a");
        let headers_b = headers_with_bearer("key-b");

        preflight(&state, &headers_a, 0).expect("first key-a request");
        let err = preflight(&state, &headers_a, 0).expect_err("second key-a request should fail");
        preflight(&state, &headers_b, 0).expect("first key-b request should pass");

        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.kind, "rate_limited");
        assert_eq!(err.message, "per-key rate limit exceeded");
    }

    #[test]
    fn per_key_concurrency_limit_isolated_by_api_key() {
        let state = Arc::new(AppState::new(
            ModelRouter::new(),
            ServerOptions {
                auth_enabled: true,
                auth_keys: vec!["key-a".into(), "key-b".into()],
                per_key_max_concurrent_requests: Some(1),
                ..ServerOptions::default()
            },
            Metrics::default(),
        ));
        let headers_a = headers_with_bearer("key-a");
        let headers_b = headers_with_bearer("key-b");

        let first = preflight(&state, &headers_a, 0).expect("first key-a request");
        let err = preflight(&state, &headers_a, 0).expect_err("second key-a request should fail");
        preflight(&state, &headers_b, 0).expect("first key-b request should pass");
        drop(first);
        preflight(&state, &headers_a, 0).expect("key-a permit released");

        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.kind, "too_many_requests");
        assert_eq!(err.message, "per-key concurrency limit exceeded");
    }

    fn headers_with_bearer(token: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {token}").parse().expect("header value"),
        );
        headers
    }
}
